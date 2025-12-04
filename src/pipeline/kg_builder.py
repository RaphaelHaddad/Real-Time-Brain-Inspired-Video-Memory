import uuid
import time
import json
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..core.config import PipelineConfig
from ..core.metrics import MetricsTracker
from ..core.logger import get_logger
from ..components.neo4j_handler import Neo4jHandler
from ..components.llm_injector import LLMInjector
from ..components.pre_llm_injector import PreLLMInjector
from ..components.global_refiner import GlobalRefiner
from ..components.network_info import NetworkInfoProvider
from .acs_automata import ACSAutomata
from .retriever_hybrid import HybridRetriever
from tqdm.asyncio import tqdm

logger = get_logger(__name__)

class KGBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_uuid = str(uuid.uuid4())
        self.metrics = MetricsTracker()
        self.neo4j_handler = Neo4jHandler(config.neo4j, config.kg, self.run_uuid)
        self.llm_injector = LLMInjector(config.llm_injector, config.chunking)
        self.network_info_provider = NetworkInfoProvider(self.neo4j_handler)
        self.acs_automata = ACSAutomata(self.neo4j_handler)
        self.online_retriever = None  # Initialize when needed

        # Initialize hierarchical extraction pipeline if enabled
        self.pre_llm_injector = None
        self.global_refiner = None
        if config.chunking.enabled:
            # Create LLM for pre-extraction (reuse llm_injector config)
            from langchain_openai import ChatOpenAI
            pre_llm = ChatOpenAI(
                base_url=config.llm_injector.endpoint,
                api_key=config.llm_injector.api_key,
                model=config.llm_injector.model_name,
                temperature=0.2,
            )
            self.pre_llm_injector = PreLLMInjector(pre_llm, config.chunking, config.embedder, config.llm_injector)

            if config.chunking.enable_global_refinement:
                self.global_refiner = GlobalRefiner(config.llm_injector, config.chunking)

        logger.info(
            f"Initialized KG Builder with run UUID: {self.run_uuid} "
            f"(hierarchical extraction: {config.chunking.enabled})"
        )

    async def build_knowledge_graph(self, vlm_json_path: str, retrieval_schedule_path: Optional[str] = None) -> str:
        """Main entry point for KG construction pipeline"""
        logger.info(f"Starting knowledge graph construction from: {vlm_json_path}")

        # Reset logs directory for this run
        logs_dir = Path("logs")
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reset logs directory: {logs_dir.absolute()}")

        try:
            # Load VLM results
            with open(vlm_json_path, 'r') as f:
                vlm_data = json.load(f)

            results = vlm_data["results"]
            total_batches = (len(results) + self.config.kg.batch_size - 1) // self.config.kg.batch_size

            # Initialize hybrid retriever if schedule provided
            if retrieval_schedule_path:
                self.online_retriever = HybridRetriever(
                    self.config.retrieval,
                    self.neo4j_handler,
                    retrieval_schedule_path
                )
                logger.info(f"Initialized hybrid retriever with {len(self.online_retriever.retrieval_schedule)} scheduled queries")

            # Process batches
            batch_progress = tqdm(total=total_batches, desc="Processing KG batches")

            for batch_idx in range(total_batches):
                batch_start_time = time.perf_counter()
                batch_start_idx = batch_idx * self.config.kg.batch_size
                batch_end_idx = min((batch_idx + 1) * self.config.kg.batch_size, len(results))
                batch = results[batch_start_idx:batch_end_idx]

                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                           f"({len(batch)} items)")

                # Aggregate content
                aggregation_start = time.perf_counter()
                aggregated_content = "\n\n".join([
                    f"Time: {item['time']}\nContent: {item['content']}"
                    for item in batch
                ])
                aggregation_time = time.perf_counter() - aggregation_start

                # Get current video time (last item in batch)
                current_video_time = batch[-1]["time"] if batch else "00:00"
                logger.debug(f"Batch {batch_idx + 1} current_video_time: {current_video_time}")

                # Prepare per-batch injection trace file under logs/
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)
                trace_file = logs_dir / f"injection_{self.run_uuid}_batch_{batch_idx + 1}.log"
                try:
                    with open(trace_file, 'w', encoding='utf-8') as tf:
                        tf.write(f"Run UUID: {self.run_uuid}\n")
                        tf.write(f"Batch: {batch_idx + 1}/{total_batches}\n")
                        tf.write("===== INITIAL BATCH ITEMS (time + content) =====\n")
                        for i, item in enumerate(batch):
                            tf.write(f"-- Item {i} --\n")
                            tf.write(f"Time: {item.get('time','')}\n")
                            tf.write("Content:\n")
                            tf.write(item.get('content',''))
                            tf.write("\n\n")
                        tf.write("===== END INITIAL ITEMS =====\n\n")
                except Exception:
                    pass

                # Get network info
                network_info_start = time.perf_counter()
                network_info = await self.network_info_provider.get_info()

                # Integrate ACS Automata context to guide LLM
                # This provides network science metrics, central entities, and recommendations based on the previous batch state
                acs_context = self.acs_automata.get_llm_context()
                if acs_context and "not yet available" not in acs_context:
                    logger.info("Injecting ACS network context into LLM prompt")
                    network_info = f"{network_info}\n\n{acs_context}"

                network_info_time = time.perf_counter() - network_info_start

                # LLM injection with hierarchical extraction
                llm_start = time.perf_counter()
                pre_triplets = []
                global_limit = self.config.chunking.global_triplet_limit
                
                # Step 1: Pre-extraction via hierarchical chunking
                text_chunks = []
                if self.pre_llm_injector:
                    pre_start = time.perf_counter()
                    pre_triplets, text_chunks, subgraphs = await self.pre_llm_injector.extract_local_triplets(
                        aggregated_content, network_info, self.neo4j_handler, batch_idx, self.run_uuid
                    )
                    pre_time = time.perf_counter() - pre_start
                    logger.info(f"Pre-extraction: {len(pre_triplets)} triplets in {pre_time:.2f}s")

                    # Write splitter chunks and per-chunk triplets to the trace
                    try:
                        with open(trace_file, 'a', encoding='utf-8') as tf:
                            chunk_details = getattr(self.pre_llm_injector, 'last_chunk_details', []) or []
                            tf.write("===== TOKEN TEXT SPLITTER OUTPUT =====\n")
                            tf.write(f"Produced {len(chunk_details)} chunks\n\n")
                            for det in chunk_details:
                                tf.write(f"-- Chunk {det.get('chunk_index')} --\n")
                                tf.write("Chunk Text:\n")
                                tf.write(det.get('chunk_text',''))
                                tf.write("\nTriplets:\n")
                                tf.write(json.dumps(det.get('triplets', []), ensure_ascii=False, indent=2))
                                if det.get('error'):
                                    tf.write(f"\nError: {det['error']}\n")
                                tf.write("\n\n")
                            tf.write("===== END SPLITTER OUTPUT =====\n\n")
                    except Exception:
                        pass
                
                # Step 2: Global refinement (if enabled)
                if self.global_refiner and pre_triplets:
                    refine_start = time.perf_counter()
                    if self.config.llm_injector.subgraph_extraction_injection:
                        # Use instruction-based refinement with subgraphs
                        pre_triplets, operations = await self.global_refiner.refine_triplets_instruction_based(
                            pre_triplets, subgraphs, global_limit
                        )
                    else:
                        # Use legacy refinement
                        pre_triplets = await self.global_refiner.refine_triplets(
                            pre_triplets, network_info, global_limit
                        )
                    refine_time = time.perf_counter() - refine_start
                    logger.info(f"Global refinement: {len(pre_triplets)} triplets in {refine_time:.2f}s")
                    # Use refined triplets directly (skip redundant final LLM call)
                    triplets = pre_triplets
                    llm_time = time.perf_counter() - llm_start
                    logger.info(f"Using {len(triplets)} refined triplets (skipped redundant final LLM enrichment)")
                else:
                    # No global refiner: use final LLM enrichment as fallback
                    triplets = await self.llm_injector.extract_triplets(
                        aggregated_content, network_info, pre_triplets, global_limit, str(trace_file)
                    )
                    llm_time = time.perf_counter() - llm_start

                # Data cleaning
                clean_start = time.perf_counter()
                cleaned_triplets = self._clean_data(triplets)
                clean_time = time.perf_counter() - clean_start

                # Inject into Neo4j (always create chunks now - hybrid search)
                neo4j_start = time.perf_counter()
                # Log the run UUID once per batch so downstream consumers can
                # easily correlate which graph UUID the batch is being pushed to.
                logger.info(f"Pushing batch {batch_idx + 1}/{total_batches} to graph UUID: {self.run_uuid}")
                inject_timings = await self.neo4j_handler.add_batch_to_graph(
                    cleaned_triplets,
                    batch_data=batch,
                    batch_idx=batch_idx,
                    text_chunks=text_chunks,
                    operations=(locals().get('operations') if 'operations' in locals() else None)
                )
                neo4j_time = time.perf_counter() - neo4j_start
                # Validate chunk creation: log counts for quick verification that
                # chunk nodes were created and, where appropriate, have embeddings
                try:
                    chunk_counts = await self.neo4j_handler.get_chunk_counts()
                    logger.info(f"Chunk node counts: {chunk_counts}")
                except Exception as e:
                    logger.debug(f"Unable to fetch chunk counts post-injection: {e}")

                # Run ACS Automata (Computes metrics AND performs semantic pruning of redundant edges)
                # This ensures the graph is optimized before the next batch starts
                acs_start = time.perf_counter()
                acs_metrics = await self.acs_automata.update_metrics()
                acs_time = time.perf_counter() - acs_start

                # Check for online retrieval
                retrieval_metrics = []
                if self.online_retriever:
                    logger.info(f"Checking for online retrieval triggers at batch end time: {current_video_time}")
                    retrieval_start = time.perf_counter()
                    queries_run = await self.online_retriever.check_and_run_queries(current_video_time)
                    retrieval_time = time.perf_counter() - retrieval_start

                    if queries_run:
                        logger.info(f"Executed {len(queries_run)} queries in {retrieval_time:.2f}s")
                        retrieval_metrics.extend(queries_run)
                    else:
                        logger.debug(f"No queries triggered at {current_video_time}")

                # Record metrics
                batch_time = time.perf_counter() - batch_start_time
                self._record_batch_metrics(
                    batch_idx,
                    batch_time,
                    {
                        "aggregation": aggregation_time,
                        "network_info": network_info_time,
                        "llm_extraction": llm_time,
                        "data_cleaning": clean_time,
                        "neo4j_injection": neo4j_time,
                        **inject_timings,
                        "acs_metrics": acs_time
                    },
                    retrieval_metrics,
                    acs_metrics
                )

                logger.info(f"Batch {batch_idx + 1} timings - Agg: {aggregation_time:.2f}s, NetInfo: {network_info_time:.2f}s, LLM: {llm_time:.2f}s, Clean: {clean_time:.2f}s, Neo4j: {neo4j_time:.2f}s, ACS: {acs_time:.2f}s")
                logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s")
                batch_progress.update(1)

            # Final metrics and cleanup
            self.metrics.save_metrics(f"metrics/kg_{self.run_uuid}.json")
            logger.info(f"Knowledge graph construction completed successfully!")
            logger.info(f"Graph UUID: {self.run_uuid}")
            logger.info(f"Total batches processed: {total_batches}")
            if self.config.chunking.enabled:
                logger.info(f"Hierarchical extraction enabled: chunk_size={self.config.chunking.chunk_size}, "
                           f"parallel={self.config.chunking.parallel_count}")

            return self.run_uuid

        finally:
            await self.neo4j_handler.close()
            if self.online_retriever:
                await self.online_retriever.close()

    def _clean_data(self, triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Industry-standard data cleaning for graph data"""
        cleaned = []
        seen = set()

        for triplet in triplets:
            # Normalize entity names
            head = triplet.get("head", "").strip().lower()
            tail = triplet.get("tail", "").strip().lower()
            rel = triplet.get("relation", "").strip().upper()

            # Skip empty or invalid triplets
            if not head or not tail or not rel:
                continue

            # Deduplicate
            key = f"{head}|{rel}|{tail}"
            if key in seen:
                continue

            cleaned.append({
                "head": head.title(),
                "relation": rel.replace("_", " ").title(),
                "tail": tail.title(),
                "source_chunks": triplet.get("source_chunks") or []
            })
            seen.add(key)

        return cleaned

    def _record_batch_metrics(self, batch_idx: int, total_time: float, timings: Dict[str, float],
                             retrieval_metrics: List[Dict], acs_metrics: Dict[str, Any]):
        """Record comprehensive metrics for this batch"""
        batch_metrics = {
            "batch_idx": batch_idx,
            "run_uuid": self.run_uuid,
            "timestamp": time.time(),
            "total_time": total_time,
            **timings,
            "retrieval_queries": retrieval_metrics,
            "acs_metrics": acs_metrics
        }

        self.metrics.add_batch_metrics(batch_metrics)
        logger.debug(f"Batch {batch_idx} metrics: {json.dumps(batch_metrics, indent=2)}")
        # Optionally save batch-level network science metrics to a running file
        try:
            if getattr(self.config, 'saving_batch_metrics', False):
                from pathlib import Path
                out_path = Path(f"metrics/{self.run_uuid}_batch_metrics_kg.json")
                existing = []
                if out_path.exists():
                    try:
                        with open(out_path, 'r') as f:
                            existing = json.load(f)
                    except Exception:
                        existing = []

                # Minimal structured entry for network metrics
                entry = {
                    "batch_idx": batch_idx,
                    "timestamp": time.time(),
                    "total_time": total_time,
                    "network_metrics": acs_metrics
                }
                existing.append(entry)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump(existing, f, indent=2, default=str)
                logger.debug(f"Saved batch network metrics to: {out_path}")
        except Exception as e:
            logger.warning(f"Failed to save per-batch metrics: {e}")