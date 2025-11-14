"""
Benchmark Evaluator: Evaluates retrieval results using LLM
- Takes retrieval results JSON
- Uses LLM to answer questions based on retrieved context
- Uses another LLM inference to evaluate correctness
- Outputs results with statistics (accuracy, etc.)
"""

import asyncio
import json
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ..core.config import PipelineConfig, BenchmarkLLMConfig
from ..core.logger import get_logger
from ..components.prompts import build_benchmark_answer_prompt, build_benchmark_evaluation_prompt
from langchain_openai import ChatOpenAI

logger = get_logger(__name__)


class BenchmarkEvaluator:
    """Evaluates retrieval results using LLM-based answer generation and correctness evaluation"""

    def __init__(self, config: PipelineConfig):
        """Initialize benchmark evaluator with LLM config"""
        self.config = config
        
        if config.benchmark_llm is None:
            raise ValueError("benchmark_llm configuration is required for benchmark evaluation")
        
        self.benchmark_config = config.benchmark_llm
        
        # Initialize LLM client
        self.llm = ChatOpenAI(
            model_name=self.benchmark_config.model_name,
            openai_api_key=self.benchmark_config.api_key,
            openai_api_base=self.benchmark_config.endpoint,
            temperature=self.benchmark_config.temperature,
            max_tokens=self.benchmark_config.max_tokens,
        )
        
        logger.info(f"Initialized BenchmarkEvaluator with model: {self.benchmark_config.model_name}")

    async def evaluate_retrieval_results(self, retrieval_results_path: str, output_path: str) -> Dict[str, Any]:
        """
        Evaluate retrieval results from JSON file
        
        Args:
            retrieval_results_path: Path to retrieval results JSON file
            output_path: Path to save benchmark results
            
        Returns:
            Dictionary with benchmark results and statistics
        """
        logger.info(f"Loading retrieval results from: {retrieval_results_path}")
        
        # Load retrieval results
        with open(retrieval_results_path, 'r') as f:
            retrieval_results = json.load(f)
        
        logger.info(f"Loaded {len(retrieval_results)} retrieval results")
        
        # Evaluate each result
        benchmark_results = []
        evaluation_stats = {
            "total_queries": len(retrieval_results),
            "correct_answers": 0,
            "incorrect_answers": 0,
            "skipped_answers": 0,
            "accuracy": 0.0,
            "results": []
        }
        
        for idx, result in enumerate(retrieval_results, 1):
            logger.info(f"Evaluating query {idx}/{len(retrieval_results)}: {result.get('query', 'N/A')}")
            
            try:
                benchmark_result = await self._evaluate_single_result(result)
                benchmark_results.append(benchmark_result)
                
                # Update statistics
                if benchmark_result.get("is_correct") is True:
                    evaluation_stats["correct_answers"] += 1
                elif benchmark_result.get("is_correct") is False:
                    evaluation_stats["incorrect_answers"] += 1
                else:
                    evaluation_stats["skipped_answers"] += 1
                    
            except Exception as e:
                logger.error(f"Error evaluating query {idx}: {str(e)}")
                benchmark_results.append({
                    "query": result.get("query", ""),
                    "groundtruth": result.get("groundtruth", ""),
                    "retrieved_answer": "",
                    "context_summary": "",
                    "is_correct": None,
                    "error": str(e)
                })
                evaluation_stats["skipped_answers"] += 1
        
        # Calculate accuracy
        evaluable_count = evaluation_stats["correct_answers"] + evaluation_stats["incorrect_answers"]
        if evaluable_count > 0:
            evaluation_stats["accuracy"] = evaluation_stats["correct_answers"] / evaluable_count
        
        # Save results
        final_output = {
            "statistics": evaluation_stats,
            "results": benchmark_results
        }
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {output_path}")
        logger.info(f"Accuracy: {evaluation_stats['accuracy']:.2%} ({evaluation_stats['correct_answers']}/{evaluable_count})")
        
        return final_output

    async def _evaluate_single_result(self, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single retrieval result
        
        Args:
            retrieval_result: Single result from retrieval_results.json
            
        Returns:
            Dictionary with evaluation result
        """
        query = retrieval_result.get("query", "")
        groundtruth = retrieval_result.get("groundtruth", "")
        retrieval_text = retrieval_result.get("retrieval", "")
        
        # Skip if no groundtruth provided
        if not groundtruth:
            logger.warning(f"Skipping evaluation - no groundtruth for query: {query}")
            return {
                "query": query,
                "groundtruth": groundtruth,
                "retrieved_answer": "",
                "context_summary": self._extract_context_summary(retrieval_text),
                "is_correct": None,
                "note": "No groundtruth provided"
            }
        
        # Step 1: Generate answer using retrieval context
        logger.debug(f"Generating answer for query: {query}")
        generated_answer = await self._generate_answer(query, retrieval_text)
        
        # Step 2: Evaluate correctness of generated answer
        logger.debug(f"Evaluating correctness of generated answer")
        is_correct = await self._evaluate_answer_correctness(
            query, groundtruth, generated_answer, retrieval_text
        )
        
        # Extract context summary (truncated chunks)
        context_summary = self._extract_context_summary(retrieval_text)
        
        return {
            "query": query,
            "groundtruth": groundtruth,
            "retrieved_answer": generated_answer,
            "context_summary": context_summary,
            "is_correct": is_correct
        }

    async def _generate_answer(self, query: str, retrieval_context: str) -> str:
        """
        Generate answer using retrieval context
        
        Args:
            query: The question
            retrieval_context: The retrieved context from the knowledge graph
            
        Returns:
            Generated answer text
        """
        try:
            # Build prompt
            answer_prompt = build_benchmark_answer_prompt()
            
            # Invoke LLM
            response = self.llm.invoke(
                answer_prompt.format_prompt(
                    question=query,
                    context=retrieval_context
                )
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    async def _evaluate_answer_correctness(
        self,
        query: str,
        groundtruth: str,
        generated_answer: str,
        retrieval_context: str
    ) -> bool:
        """
        Evaluate if generated answer is correct
        
        Args:
            query: The question
            groundtruth: Expected correct answer
            generated_answer: LLM-generated answer
            retrieval_context: The context used
            
        Returns:
            Boolean indicating correctness
        """
        try:
            # Build prompt
            eval_prompt = build_benchmark_evaluation_prompt()
            
            # Invoke LLM
            response = self.llm.invoke(
                eval_prompt.format_prompt(
                    question=query,
                    groundtruth=groundtruth,
                    generated_answer=generated_answer,
                    context=retrieval_context
                )
            )
            
            # Parse response (should be JSON)
            response_text = response.content.strip()
            
            # Try to extract JSON
            try:
                result_dict = json.loads(response_text)
                is_correct = result_dict.get("is_correct", False)
                return bool(is_correct)
            except json.JSONDecodeError:
                # Fallback: check if response contains "true" or "false"
                logger.warning(f"Could not parse evaluation response as JSON: {response_text}")
                return "true" in response_text.lower()
                
        except Exception as e:
            logger.error(f"Error evaluating answer correctness: {str(e)}")
            raise

    @staticmethod
    def _extract_context_summary(retrieval_text: str, max_length: int = 300) -> str:
        """
        Extract a summary of the context from retrieval text
        
        Args:
            retrieval_text: Full retrieval text
            max_length: Maximum length of summary
            
        Returns:
            Truncated context summary
        """
        # Remove "Found results for..." prefix and extract content
        if retrieval_text.startswith("Found results for"):
            # Extract just the chunk descriptions
            lines = retrieval_text.split('\n')
            content_lines = [l for l in lines if l.strip() and not l.startswith("Found results")]
            retrieval_text = '\n'.join(content_lines)
        
        # Truncate to max_length
        if len(retrieval_text) > max_length:
            return retrieval_text[:max_length] + "..."
        return retrieval_text
