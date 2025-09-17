"""
Model Optimization Toolkit for Plant Disease Detection
Provides various techniques to make models lightweight:
1. TensorFlow Lite Quantization
2. Model Pruning
3. Knowledge Distillation
4. Efficient Architecture Design
"""

import tensorflow as tf
import numpy as np
import os
import time
import json
from pathlib import Path
import shutil


class ModelOptimizer:
    def __init__(self, model_path='best_model.h5', class_names_path='class_names.txt'):
        """
        Initialize the model optimizer
        
        Args:
            model_path (str): Path to the original model
            class_names_path (str): Path to class names file
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = []
        
        # Load model and class names
        self.load_original_model()
        self.load_class_names()
        
        # Create output directory for optimized models
        self.output_dir = Path('optimized_models')
        self.output_dir.mkdir(exist_ok=True)
    
    def load_original_model(self):
        """Load the original model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"✅ Loaded original model from {self.model_path}")
            print(f"📊 Model size: {self.get_model_size(self.model_path):.2f} MB")
            print(f"🧠 Parameters: {self.model.count_params():,}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_class_names(self):
        """Load class names"""
        try:
            with open(self.class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.class_names)} class names")
        except Exception as e:
            print(f"❌ Error loading class names: {e}")
            raise
    
    def get_model_size(self, model_path):
        """Get model file size in MB"""
        return os.path.getsize(model_path) / (1024 * 1024)
    
    def create_representative_dataset(self, num_samples=100):
        """
        Create a representative dataset for quantization
        This is a dummy dataset since we don't have access to training data
        """
        def representative_data_gen():
            for _ in range(num_samples):
                # Generate random data with the same shape as model input
                input_shape = self.model.input_shape[1:]  # Remove batch dimension
                data = np.random.random((1, *input_shape)).astype(np.float32)
                yield [data]
        
        return representative_data_gen
    
    def quantize_model_int8(self, output_name='model_quantized_int8.tflite'):
        """
        Apply INT8 quantization for maximum compression
        """
        print("\n🔄 Starting INT8 quantization...")
        
        try:
            # Convert to TensorFlow Lite with INT8 quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.create_representative_dataset()
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            tflite_quantized_model = converter.convert()
            
            # Save the quantized model
            output_path = self.output_dir / output_name
            with open(output_path, 'wb') as f:
                f.write(tflite_quantized_model)
            
            # Get size comparison
            original_size = self.get_model_size(self.model_path)
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
            compression_ratio = (1 - quantized_size / original_size) * 100
            
            print(f"✅ INT8 quantized model saved to {output_path}")
            print(f"📊 Original size: {original_size:.2f} MB")
            print(f"📊 Quantized size: {quantized_size:.2f} MB")
            print(f"🎯 Compression: {compression_ratio:.1f}% reduction")
            
            return str(output_path), quantized_size
            
        except Exception as e:
            print(f"❌ Error during INT8 quantization: {e}")
            return None, None
    
    def quantize_model_float16(self, output_name='model_quantized_float16.tflite'):
        """
        Apply Float16 quantization for balanced compression and accuracy
        """
        print("\n🔄 Starting Float16 quantization...")
        
        try:
            # Convert to TensorFlow Lite with Float16 quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_quantized_model = converter.convert()
            
            # Save the quantized model
            output_path = self.output_dir / output_name
            with open(output_path, 'wb') as f:
                f.write(tflite_quantized_model)
            
            # Get size comparison
            original_size = self.get_model_size(self.model_path)
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
            compression_ratio = (1 - quantized_size / original_size) * 100
            
            print(f"✅ Float16 quantized model saved to {output_path}")
            print(f"📊 Original size: {original_size:.2f} MB")
            print(f"📊 Quantized size: {quantized_size:.2f} MB")
            print(f"🎯 Compression: {compression_ratio:.1f}% reduction")
            
            return str(output_path), quantized_size
            
        except Exception as e:
            print(f"❌ Error during Float16 quantization: {e}")
            return None, None
    
    def create_pruned_model(self, output_name='model_pruned.h5', sparsity=0.5):
        """
        Create a pruned version of the model
        Note: This is a simplified pruning approach
        """
        print(f"\n🔄 Starting model pruning with {sparsity*100}% sparsity...")
        
        try:
            import tensorflow_model_optimization as tfmot
            
            # Define pruning schedule
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=sparsity,
                begin_step=0,
                end_step=1000
            )
            
            # Apply pruning to dense layers only
            def apply_pruning_to_dense(layer):
                if isinstance(layer, tf.keras.layers.Dense):
                    return tfmot.sparsity.keras.prune_low_magnitude(
                        layer, pruning_schedule=pruning_schedule
                    )
                return layer
            
            # Clone and prune the model
            pruned_model = tf.keras.models.clone_model(
                self.model,
                clone_function=apply_pruning_to_dense,
            )
            
            # Compile the pruned model
            pruned_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save the pruned model
            output_path = self.output_dir / output_name
            pruned_model.save(output_path)
            
            print(f"✅ Pruned model saved to {output_path}")
            print(f"🧠 Pruned model parameters: {pruned_model.count_params():,}")
            
            return str(output_path)
            
        except ImportError:
            print("⚠️ TensorFlow Model Optimization toolkit not available")
            print("Install with: pip install tensorflow-model-optimization")
            return None
        except Exception as e:
            print(f"❌ Error during model pruning: {e}")
            return None
    
    def create_lightweight_architecture(self, num_classes=None):
        """
        Create a new lightweight model architecture optimized for inference
        """
        if num_classes is None:
            num_classes = len(self.class_names)
        
        print(f"\n🔄 Creating lightweight model architecture for {num_classes} classes...")
        
        try:
            # Try MobileNetV3Small first, then fallback to MobileNetV2 with smaller alpha
            try:
                # Use MobileNetV3Small as base (much smaller than MobileNetV2)
                base_model = tf.keras.applications.MobileNetV3Small(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    alpha=1.0,  # Use 1.0 for minimalistic
                    minimalistic=True  # Use minimalistic architecture
                )
            except Exception as e:
                print(f"⚠️ MobileNetV3Small failed ({e}), using MobileNetV2 with alpha=0.5")
                # Fallback to MobileNetV2 with smaller width multiplier
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    alpha=0.5  # Much smaller width multiplier
                )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add efficient classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),  # Smaller dense layer
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save the lightweight model
            output_path = self.output_dir / 'model_lightweight_architecture.h5'
            model.save(output_path)
            
            print(f"✅ Lightweight model saved to {output_path}")
            print(f"🧠 Parameters: {model.count_params():,}")
            print(f"📊 Estimated size: {model.count_params() * 4 / (1024 * 1024):.2f} MB")
            
            return str(output_path), model
            
        except Exception as e:
            print(f"❌ Error creating lightweight architecture: {e}")
            return None, None
    
    def benchmark_inference_speed(self, model_path, num_runs=100):
        """
        Benchmark inference speed of a model
        """
        print(f"\n⏱️ Benchmarking inference speed for {Path(model_path).name}...")
        
        try:
            if model_path.endswith('.tflite'):
                # TensorFlow Lite model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Generate test input with correct data type
                input_shape = input_details[0]['shape']
                input_dtype = input_details[0]['dtype']
                
                if input_dtype == np.uint8:
                    # For INT8 quantized models
                    test_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
                else:
                    # For Float models
                    test_input = np.random.random(input_shape).astype(input_dtype)
                
                # Warm up
                for _ in range(10):
                    interpreter.set_tensor(input_details[0]['index'], test_input)
                    interpreter.invoke()
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_runs):
                    interpreter.set_tensor(input_details[0]['index'], test_input)
                    interpreter.invoke()
                end_time = time.time()
                
            else:
                # Regular Keras model
                model = tf.keras.models.load_model(model_path, compile=False)
                test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                
                # Warm up
                for _ in range(10):
                    model.predict(test_input, verbose=0)
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_runs):
                    model.predict(test_input, verbose=0)
                end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
            fps = 1000 / avg_time
            
            print(f"⚡ Average inference time: {avg_time:.2f} ms")
            print(f"🎬 Approximate FPS: {fps:.1f}")
            
            return avg_time, fps
            
        except Exception as e:
            print(f"❌ Error benchmarking: {e}")
            return None, None
    
    def optimize_all_models(self):
        """
        Run all optimization techniques and create a comparison report
        """
        print("🚀 Starting comprehensive model optimization...")
        
        results = {
            'original': {
                'path': self.model_path,
                'size_mb': self.get_model_size(self.model_path),
                'parameters': self.model.count_params()
            }
        }
        
        # 1. INT8 Quantization
        int8_path, int8_size = self.quantize_model_int8()
        if int8_path:
            int8_time, int8_fps = self.benchmark_inference_speed(int8_path)
            results['int8_quantized'] = {
                'path': int8_path,
                'size_mb': int8_size,
                'inference_time_ms': int8_time,
                'fps': int8_fps
            }
        
        # 2. Float16 Quantization
        float16_path, float16_size = self.quantize_model_float16()
        if float16_path:
            float16_time, float16_fps = self.benchmark_inference_speed(float16_path)
            results['float16_quantized'] = {
                'path': float16_path,
                'size_mb': float16_size,
                'inference_time_ms': float16_time,
                'fps': float16_fps
            }
        
        # 3. Lightweight Architecture
        lightweight_path, lightweight_model = self.create_lightweight_architecture()
        if lightweight_path:
            lightweight_size = self.get_model_size(lightweight_path)
            lightweight_time, lightweight_fps = self.benchmark_inference_speed(lightweight_path)
            results['lightweight_architecture'] = {
                'path': lightweight_path,
                'size_mb': lightweight_size,
                'parameters': lightweight_model.count_params(),
                'inference_time_ms': lightweight_time,
                'fps': lightweight_fps
            }
        
        # 4. Original model benchmark
        original_time, original_fps = self.benchmark_inference_speed(self.model_path)
        results['original']['inference_time_ms'] = original_time
        results['original']['fps'] = original_fps
        
        # Save results
        results_path = self.output_dir / 'optimization_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print comparison table
        self.print_comparison_table(results)
        
        return results
    
    def print_comparison_table(self, results):
        """
        Print a nice comparison table of all optimized models
        """
        print("\n" + "="*80)
        print("📊 MODEL OPTIMIZATION COMPARISON")
        print("="*80)
        
        print(f"{'Model Type':<25} {'Size (MB)':<12} {'Parameters':<12} {'Time (ms)':<12} {'FPS':<8}")
        print("-" * 80)
        
        for model_type, data in results.items():
            size = f"{data['size_mb']:.2f}"
            params = f"{data.get('parameters', 'N/A'):,}" if isinstance(data.get('parameters'), int) else 'N/A'
            time_ms = f"{data.get('inference_time_ms', 0):.2f}" if data.get('inference_time_ms') else 'N/A'
            fps = f"{data.get('fps', 0):.1f}" if data.get('fps') else 'N/A'
            
            print(f"{model_type.replace('_', ' ').title():<25} {size:<12} {params:<12} {time_ms:<12} {fps:<8}")
        
        print("="*80)
        
        # Show best options
        if 'int8_quantized' in results:
            original_size = results['original']['size_mb']
            int8_size = results['int8_quantized']['size_mb']
            reduction = (1 - int8_size / original_size) * 100
            print(f"🏆 Best compression: INT8 quantization ({reduction:.1f}% size reduction)")
        
        if 'lightweight_architecture' in results:
            original_params = results['original']['parameters']
            lightweight_params = results['lightweight_architecture']['parameters']
            param_reduction = (1 - lightweight_params / original_params) * 100
            print(f"🏆 Most efficient: Lightweight architecture ({param_reduction:.1f}% parameter reduction)")


def main():
    """
    Main function to run model optimization
    """
    try:
        # Initialize optimizer
        optimizer = ModelOptimizer()
        
        # Run all optimizations
        results = optimizer.optimize_all_models()
        
        print("\n✅ Model optimization completed!")
        print(f"📁 Optimized models saved in: {optimizer.output_dir}")
        print("📋 Results saved in: optimization_results.json")
        
        # Recommend best model
        print("\n💡 RECOMMENDATIONS:")
        print("- For mobile deployment: Use INT8 quantized model (smallest size)")
        print("- For edge devices: Use Float16 quantized model (balanced size/accuracy)")
        print("- For new training: Use lightweight architecture")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")


if __name__ == "__main__":
    main()