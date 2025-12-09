#!/usr/bin/env python3
"""
Main CLI entry point for the VidGraph pipeline
"""
import asyncio
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import PipelineConfig
from src.core.logger import get_logger

logger = get_logger(__name__)

async def run_vlm_extraction(config_path: str, video_path: str, output_path: str):
    """Run the VLM extraction pipeline"""
    from src.pipeline.vlm_extractor import VLMExtractor

    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)

    logger.info(f"Processing video: {video_path}")
    extractor = VLMExtractor(config)

    result_path = await extractor.process_video(video_path, output_path)
    logger.info(f"VLM extraction completed: {result_path}")

    return result_path

async def run_kg_construction(config_path: str, vlm_json_path: str, retrieval_schedule_path: str = None):
    """Run the knowledge graph construction pipeline"""
    from src.pipeline.kg_builder import KGBuilder

    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)

    logger.info(f"Building knowledge graph from: {vlm_json_path}")
    if retrieval_schedule_path:
        logger.info(f"Using retrieval schedule: {retrieval_schedule_path}")

    builder = KGBuilder(config)
    graph_uuid = await builder.build_knowledge_graph(vlm_json_path, retrieval_schedule_path)
    logger.info(f"Knowledge graph construction completed with UUID: {graph_uuid}")

    return graph_uuid

async def run_offline_retrieval(config_path: str, graph_uuid: str, query: str, groundtruth: str = "", true_chunks: list = None):
    """Run offline retrieval against a specific graph"""
    from src.pipeline.retriever import OfflineRetriever

    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)

    logger.info(f"Performing offline retrieval on graph: {graph_uuid}")
    logger.info(f"Query: {query}")
    logger.info(f"Ground truth: {groundtruth}")

    retriever = OfflineRetriever(config.retrieval, config.neo4j, config.kg, config.community_high_graph)
    result = await retriever.retrieve(query, graph_uuid, groundtruth, true_chunks)

    logger.info(f"Retrieval result: {result}")
    return result

async def run_batch_offline_retrieval(config_path: str, graph_uuid: str, input_file: str, output_file: str, expected_chunk_json: str = None, deep_analysis_dir: str = None):
    """Run batch offline retrieval from a JSON file"""
    from src.pipeline.retriever import OfflineRetriever

    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)

    logger.info(f"Performing batch offline retrieval on graph: {graph_uuid}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    retriever = OfflineRetriever(config.retrieval, config.neo4j, config.kg, config.community_high_graph)
    results = await retriever.batch_retrieve_from_file(input_file, graph_uuid, expected_chunk_json, deep_analysis_dir)

    # Save results to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Batch retrieval completed with {len(results)} results")

    # Save retrieval timing metrics to metrics/<graph_uuid>_retrieval_times_<timestamp>.json
    try:
        metrics_dir = Path('metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        metrics_file = metrics_dir / f"retrieval_times_{graph_uuid}_{timestamp}.json"

        # Prepare metrics summary
        per_query = []
        total_time = 0.0
        for item in results:
            rt = item.get('retrieval_time', 0.0) or 0.0
            per_query.append({
                'query': item.get('query', ''),
                'groundtruth': item.get('groundtruth', ''),
                'retrieval_time': rt
            })
            total_time += float(rt)

        avg_time = total_time / len(results) if results else 0.0
        metrics_data = {
            'graph_uuid': graph_uuid,
            'created_at': timestamp,
            'total_queries': len(results),
            'total_time': total_time,
            'average_time': avg_time,
            'per_query': per_query
        }

        with open(metrics_file, 'w', encoding='utf-8') as mf:
            json.dump(metrics_data, mf, indent=2)

        logger.info(f"Saved retrieval timing metrics to: {metrics_file}")
    except Exception as e:
        logger.warning(f"Failed to save retrieval metrics: {e}")
    return output_file

async def run_export_graph(config_path: str, graph_uuid: str, output_path: str):
    """Export a knowledge graph for collaboration"""
    from src.components.graph_exporter import GraphExporter
    from src.components.neo4j_handler import Neo4jHandler

    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)
    
    logger.info(f"Exporting graph with UUID: {graph_uuid}")
    
    neo4j_handler = Neo4jHandler(config.neo4j, config.kg, graph_uuid)
    exporter = GraphExporter(neo4j_handler)
    
    try:
        result_path = await exporter.export_graph(graph_uuid, output_path)
        logger.info(f"Graph export completed: {result_path}")
        return result_path
    finally:
        await neo4j_handler.close()

async def run_import_graph(config_path: str, input_path: str, new_uuid: str = None):
    """Import a knowledge graph from an exported file"""
    from src.components.graph_exporter import GraphImporter
    from src.components.neo4j_handler import Neo4jHandler

    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)
    
    logger.info(f"Importing graph from: {input_path}")
    if new_uuid:
        logger.info(f"Using new UUID: {new_uuid}")
    
    neo4j_handler = Neo4jHandler(config.neo4j, config.kg, new_uuid or "temp")
    importer = GraphImporter(neo4j_handler)
    
    try:
        final_uuid = await importer.import_graph(input_path, new_uuid)
        logger.info(f"Graph import completed with UUID: {final_uuid}")
        return final_uuid
    finally:
        await neo4j_handler.close()

async def run_benchmark(config_path: str, input_file: str, output_file: str):
    """Run benchmark evaluation on retrieval results"""
    from src.pipeline.benchmark import BenchmarkEvaluator

    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)
    
    logger.info(f"Running benchmark evaluation on: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    evaluator = BenchmarkEvaluator(config)
    results = await evaluator.evaluate_retrieval_results(input_file, output_file)
    
    logger.info(f"Benchmark completed with accuracy: {results['statistics']['accuracy']:.2%}")
    return output_file


async def run_precompute(config_path: str, graph_uuid: str, methods: list):
    """Run pre-retrieval computation (PageRank, CH3-L3) on a knowledge graph"""
    from src.pipeline.precompute import run_precomputation

    logger.info(f"Loading configuration from: {config_path}")
    logger.info(f"Running precomputation for graph: {graph_uuid}")
    logger.info(f"Methods: {methods}")

    results = await run_precomputation(config_path, graph_uuid, methods)
    
    for method, stats in results.items():
        if stats.get("status") == "success":
            logger.info(f"✓ {method}: {stats.get('nodes_processed', 0)} nodes processed in {stats.get('elapsed_seconds', 0):.2f}s")
        else:
            logger.warning(f"✗ {method}: {stats.get('status', 'unknown')} - {stats.get('reason', stats.get('message', ''))}")
    
    return results


async def check_precomputation_for_retrieval(config_path: str, graph_uuid: str) -> bool:
    """Check if required precomputation exists for the configured hop method"""
    from src.components.neo4j_handler import Neo4jHandler
    
    config = PipelineConfig.from_yaml(config_path)
    hop_method = config.retrieval.hop_method
    
    # Naive method doesn't require precomputation
    if hop_method == "naive":
        return True
    
    # Check if precomputation exists
    neo4j_handler = Neo4jHandler(config.neo4j, config.kg, graph_uuid)
    try:
        has_precompute = await neo4j_handler.has_precomputation(graph_uuid, hop_method)
        
        if not has_precompute:
            logger.error(f"❌ Retrieval requires '{hop_method}' precomputation, but graph {graph_uuid} has not been precomputed.")
            logger.error(f"   Run: python3 -m src.cli.main precompute --config {config_path} --graph-uuid {graph_uuid} --methods {hop_method}")
            return False
        
        return True
    finally:
        await neo4j_handler.close()


def main():
    parser = argparse.ArgumentParser(description="VidGraph: Video-to-Knowledge Graph Pipeline")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # VLM extraction command
    vlm_parser = subparsers.add_parser('vlm', help='Run VLM extraction on a video')
    vlm_parser.add_argument('--config', required=True, help='Path to configuration file')
    vlm_parser.add_argument('--video', required=True, help='Path to input video file')
    vlm_parser.add_argument('--output', required=True, help='Path for output JSON file')

    # KG construction command
    kg_parser = subparsers.add_parser('kg', help='Build knowledge graph from VLM output')
    kg_parser.add_argument('--config', required=True, help='Path to configuration file')
    kg_parser.add_argument('--vlm-output', required=True, help='Path to VLM output JSON file')
    kg_parser.add_argument('--retrieval-schedule', help='Path to retrieval schedule JSON file (optional)')

    # Offline retrieval command
    retrieval_parser = subparsers.add_parser('retrieve', help='Run offline retrieval')
    retrieval_parser.add_argument('--config', required=True, help='Path to configuration file')
    retrieval_parser.add_argument('--graph-uuid', required=True, help='UUID of the knowledge graph to query')
    retrieval_parser.add_argument('--query', required=True, help='Query to execute against the graph')
    retrieval_parser.add_argument('--groundtruth', help='Ground truth answer for evaluation (optional)')
    retrieval_parser.add_argument('--true_chunks', nargs='*', help='Optional list of true chunk indices (e.g. 2 6 40 or 2,6,40)')

    # Batch offline retrieval command
    batch_retrieval_parser = subparsers.add_parser('batch-retrieve', help='Run batch offline retrieval from JSON file')
    batch_retrieval_parser.add_argument('--config', required=True, help='Path to configuration file')
    batch_retrieval_parser.add_argument('--graph-uuid', required=True, help='UUID of the knowledge graph to query')
    batch_retrieval_parser.add_argument('--input', required=True, help='Path to input JSON file with queries')
    batch_retrieval_parser.add_argument('--output', required=True, help='Path for output JSON file with results')
    batch_retrieval_parser.add_argument('--expected-chunk-json', help='Path to expected chunk JSON to enable deep analysis logging')
    batch_retrieval_parser.add_argument('--deep-analysis-dir', help='(Optional) Directory to write deep-analysis per-query JSON files. Requires --expected-chunk-json.')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export a knowledge graph for collaboration')
    export_parser.add_argument('--config', required=True, help='Path to configuration file')
    export_parser.add_argument('--graph-uuid', required=True, help='UUID of the graph to export')
    export_parser.add_argument('--output', required=True, help='Path for exported JSON file')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import a knowledge graph from exported file')
    import_parser.add_argument('--config', required=True, help='Path to configuration file')
    import_parser.add_argument('--input', required=True, help='Path to imported JSON file')
    import_parser.add_argument('--new-uuid', help='New UUID for the imported graph (optional, will generate new one if not provided)')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark evaluation on retrieval results')
    benchmark_parser.add_argument('--config', required=True, help='Path to configuration file')
    benchmark_parser.add_argument('--input', required=True, help='Path to retrieval results JSON file')
    benchmark_parser.add_argument('--output', required=True, help='Path for benchmark results JSON file')

    # Precompute command
    precompute_parser = subparsers.add_parser('precompute', help='Run pre-retrieval computation (PageRank, CH3-L3)')
    precompute_parser.add_argument('--config', required=True, help='Path to configuration file')
    precompute_parser.add_argument('--graph-uuid', required=True, help='UUID of the knowledge graph')
    precompute_parser.add_argument('--methods', nargs='+', required=True, 
                                   choices=['page_rank', 'ch3_l3'],
                                   help='Methods to precompute (page_rank, ch3_l3)')

    args = parser.parse_args()

    if args.command == 'vlm':
        result = asyncio.run(run_vlm_extraction(args.config, args.video, args.output))
        print(f"VLM extraction completed: {result}")

    elif args.command == 'kg':
        retrieval_schedule = args.retrieval_schedule if hasattr(args, 'retrieval_schedule') else None
        result = asyncio.run(run_kg_construction(args.config, args.vlm_output, retrieval_schedule))
        print(f"Knowledge graph construction completed with UUID: {result}")

    elif args.command == 'retrieve':
        # Check if precomputation is required and exists
        if not asyncio.run(check_precomputation_for_retrieval(args.config, args.graph_uuid)):
            sys.exit(1)
        
        # Parse true_chunks into a list of ints if provided
        true_chunks_arg = getattr(args, 'true_chunks', None)
        true_chunks = None
        if true_chunks_arg:
            # Support '2 6 40' or '2,6,40' style inputs
            parsed = []
            for part in true_chunks_arg:
                # Split commas if used
                if isinstance(part, str) and ',' in part:
                    parsed.extend([p.strip() for p in part.split(',') if p.strip()])
                else:
                    parsed.append(part)
            try:
                true_chunks = [int(x) for x in parsed]
            except Exception:
                logger.warning("Could not parse --true_chunks; expected space-separated integers or comma-separated list. Ignoring true_chunks argument.")
                true_chunks = None

        result = asyncio.run(run_offline_retrieval(args.config, args.graph_uuid, args.query, args.groundtruth, true_chunks))
        print(f"Retrieval completed: {result}")

    elif args.command == 'batch-retrieve':
        # Check if precomputation is required and exists
        if not asyncio.run(check_precomputation_for_retrieval(args.config, args.graph_uuid)):
            sys.exit(1)
        # Protect deep-analysis dir: only allowed if expected_chunk_json is provided
        if getattr(args, 'deep_analysis_dir', None) and not getattr(args, 'expected_chunk_json', None):
            logger.error("--deep-analysis-dir can only be used together with --expected-chunk-json")
            sys.exit(1)

        result = asyncio.run(run_batch_offline_retrieval(
            args.config,
            args.graph_uuid,
            args.input,
            args.output,
            getattr(args, 'expected_chunk_json', None),
            getattr(args, 'deep_analysis_dir', None)
        ))
        print(f"Batch retrieval completed: {result}")

    elif args.command == 'export':
        result = asyncio.run(run_export_graph(args.config, args.graph_uuid, args.output))
        print(f"Graph export completed: {result}")

    elif args.command == 'import':
        result = asyncio.run(run_import_graph(args.config, args.input, args.new_uuid))
        print(f"Graph import completed with UUID: {result}")

    elif args.command == 'benchmark':
        result = asyncio.run(run_benchmark(args.config, args.input, args.output))
        print(f"Benchmark evaluation completed: {result}")

    elif args.command == 'precompute':
        result = asyncio.run(run_precompute(args.config, args.graph_uuid, args.methods))
        print(f"Precomputation completed: {json.dumps(result, indent=2)}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()