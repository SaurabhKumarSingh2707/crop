"""
Model Performance Comparison Utility
Compare inference speed, memory usage, and accuracy across different model versions
"""

import time
import psutil
import os
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

# Import our predictors
from predict_advanced import AdvancedPlantDiseasePredictor
from predict_lightweight import LightweightPlantDiseasePredictor


class ModelBenchmark:
    def __init__(self):
        """Initialize the benchmark suite"""
        self.results = {}
        self.test_image = self.create_test_image()
    
    def create_test_image(self):
        """Create a consistent test image for all benchmarks"""
        np.random.seed(42)  # For reproducible results
        return np.random.random((224, 224, 3)).astype(np.float32)
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def benchmark_model(self, predictor, model_name, num_runs=50):
        """
        Benchmark a specific model
        
        Args:
            predictor: The predictor instance
            model_name (str): Name of the model for logging
            num_runs (int): Number of inference runs for averaging
        """
        print(f"\n🔄 Benchmarking {model_name}...")
        
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Get model info
        model_info = predictor.get_model_info()
        
        # Warm up runs
        print("  🔥 Warming up...")
        for _ in range(5):
            try:
                predictor.predict_image_from_array(self.test_image, top_n=1)
            except Exception as e:
                print(f"  ❌ Warmup failed: {e}")
                return None
        
        # Memory usage after warmup
        warmed_memory = self.get_memory_usage()
        
        # Benchmark inference speed
        print(f"  ⚡ Running {num_runs} inference tests...")
        times = []
        predictions = []
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = predictor.predict_image_from_array(self.test_image, top_n=5)
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(inference_time)
                predictions.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"    Completed {i + 1}/{num_runs} runs")
                    
            except Exception as e:
                print(f"  ❌ Inference {i+1} failed: {e}")
                continue
        
        if not times:
            print(f"  ❌ All inferences failed for {model_name}")
            return None
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        
        # Memory usage
        memory_used = warmed_memory - initial_memory
        
        # Model size
        model_path = model_info.get('model_path', '')
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            model_size_mb = float(model_info.get('model_size_mb', 0))
        
        # Get prediction consistency (check if predictions are consistent)
        top_predictions = [p['top_prediction'] for p in predictions if 'top_prediction' in p]
        prediction_consistency = len(set(top_predictions)) / len(top_predictions) if top_predictions else 0
        
        benchmark_result = {
            'model_name': model_name,
            'model_info': model_info,
            'performance': {
                'avg_inference_time_ms': round(avg_time, 3),
                'std_inference_time_ms': round(std_time, 3),
                'min_inference_time_ms': round(min_time, 3),
                'max_inference_time_ms': round(max_time, 3),
                'fps': round(fps, 2),
                'successful_runs': len(times),
                'total_runs': num_runs,
                'success_rate': len(times) / num_runs * 100
            },
            'memory': {
                'initial_mb': round(initial_memory, 2),
                'warmed_mb': round(warmed_memory, 2),
                'used_mb': round(memory_used, 2)
            },
            'model_specs': {
                'size_mb': round(model_size_mb, 2),
                'type': model_info.get('model_type', 'unknown'),
                'quantized': model_info.get('is_quantized', False),
                'optimized': model_info.get('inference_optimized', False)
            },
            'prediction_consistency': round(prediction_consistency, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  ✅ {model_name} benchmark completed")
        print(f"     Avg time: {avg_time:.2f}ms | FPS: {fps:.1f} | Size: {model_size_mb:.2f}MB")
        
        return benchmark_result
    
    def run_comprehensive_benchmark(self):
        """Run benchmarks on all available models"""
        print("🚀 Starting comprehensive model benchmark...")
        
        models_to_test = []
        
        # Test original model
        try:
            predictor = AdvancedPlantDiseasePredictor(model_path='best_model.h5')
            models_to_test.append((predictor, 'Original MobileNetV2'))
        except Exception as e:
            print(f"⚠️ Could not load original model: {e}")
        
        # Test lightweight architecture
        try:
            if os.path.exists('optimized_models/model_lightweight_architecture.h5'):
                predictor = LightweightPlantDiseasePredictor(
                    model_path='optimized_models/model_lightweight_architecture.h5'
                )
                models_to_test.append((predictor, 'Lightweight MobileNetV3'))
        except Exception as e:
            print(f"⚠️ Could not load lightweight model: {e}")
        
        # Test Float16 quantized model
        try:
            if os.path.exists('optimized_models/model_quantized_float16.tflite'):
                predictor = LightweightPlantDiseasePredictor(
                    model_path='optimized_models/model_quantized_float16.tflite'
                )
                models_to_test.append((predictor, 'Float16 Quantized'))
        except Exception as e:
            print(f"⚠️ Could not load Float16 model: {e}")
        
        # Test INT8 quantized model
        try:
            if os.path.exists('optimized_models/model_quantized_int8.tflite'):
                predictor = LightweightPlantDiseasePredictor(
                    model_path='optimized_models/model_quantized_int8.tflite'
                )
                models_to_test.append((predictor, 'INT8 Quantized'))
        except Exception as e:
            print(f"⚠️ Could not load INT8 model: {e}")
        
        # Run benchmarks
        for predictor, model_name in models_to_test:
            result = self.benchmark_model(predictor, model_name)
            if result:
                self.results[model_name] = result
        
        # Save results
        self.save_results()
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to JSON file"""
        output_file = 'model_benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📁 Results saved to {output_file}")
    
    def generate_comparison_report(self):
        """Generate a detailed comparison report"""
        if not self.results:
            print("❌ No results to compare")
            return
        
        print("\\n" + "="*80)
        print("📊 COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        # Table header
        print(f"{'Model':<20} {'Size(MB)':<10} {'Avg Time(ms)':<12} {'FPS':<8} {'Memory(MB)':<12} {'Type':<15}")
        print("-" * 80)
        
        # Sort by inference time (fastest first)
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['performance']['avg_inference_time_ms']
        )
        
        for model_name, result in sorted_results:
            size = result['model_specs']['size_mb']
            avg_time = result['performance']['avg_inference_time_ms']
            fps = result['performance']['fps']
            memory = result['memory']['used_mb']
            model_type = result['model_specs']['type']
            
            print(f"{model_name:<20} {size:<10.2f} {avg_time:<12.2f} {fps:<8.1f} {memory:<12.2f} {model_type:<15}")
        
        print("="*80)
        
        # Performance rankings
        print("\\n🏆 PERFORMANCE RANKINGS:")
        
        # Fastest inference
        fastest = min(self.results.items(), key=lambda x: x[1]['performance']['avg_inference_time_ms'])
        print(f"⚡ Fastest inference: {fastest[0]} ({fastest[1]['performance']['avg_inference_time_ms']:.2f}ms)")
        
        # Smallest model
        smallest = min(self.results.items(), key=lambda x: x[1]['model_specs']['size_mb'])
        print(f"📦 Smallest model: {smallest[0]} ({smallest[1]['model_specs']['size_mb']:.2f}MB)")
        
        # Most memory efficient
        most_efficient = min(self.results.items(), key=lambda x: x[1]['memory']['used_mb'])
        print(f"🧠 Most memory efficient: {most_efficient[0]} ({most_efficient[1]['memory']['used_mb']:.2f}MB)")
        
        # Calculate improvements
        if 'Original MobileNetV2' in self.results:
            original = self.results['Original MobileNetV2']
            print("\\n📈 IMPROVEMENTS OVER ORIGINAL:")
            
            for model_name, result in self.results.items():
                if model_name == 'Original MobileNetV2':
                    continue
                
                speed_improvement = (original['performance']['avg_inference_time_ms'] / 
                                   result['performance']['avg_inference_time_ms'] - 1) * 100
                size_reduction = (1 - result['model_specs']['size_mb'] / 
                                original['model_specs']['size_mb']) * 100
                
                print(f"  {model_name}:")
                print(f"    Speed: {speed_improvement:+.1f}% {'faster' if speed_improvement > 0 else 'slower'}")
                print(f"    Size: {size_reduction:.1f}% smaller")
        
        # Recommendations
        print("\\n💡 RECOMMENDATIONS:")
        
        # Find best overall model (balanced speed and size)
        best_overall = None
        best_score = float('inf')
        
        for model_name, result in self.results.items():
            # Score based on normalized inference time and model size
            time_score = result['performance']['avg_inference_time_ms'] / 100  # Normalize
            size_score = result['model_specs']['size_mb'] / 10  # Normalize
            total_score = time_score + size_score
            
            if total_score < best_score:
                best_score = total_score
                best_overall = model_name
        
        if best_overall:
            print(f"🥇 Best overall: {best_overall} (balanced speed and size)")
        
        print("\\n🎯 Use case recommendations:")
        print("- Mobile/Edge deployment: Choose smallest quantized model")
        print("- Real-time applications: Choose fastest model")
        print("- Production servers: Choose best overall model")
        print("- Development/Testing: Original model for highest accuracy")


def main():
    """Main function to run comprehensive benchmarks"""
    try:
        benchmark = ModelBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        print("\\n✅ Benchmark completed successfully!")
        print(f"📊 Tested {len(results)} models")
        print("📋 Detailed results saved to model_benchmark_results.json")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()