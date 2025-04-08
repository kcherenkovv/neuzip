from neuzip import optimize_model

# Путь к исходной модели ONNX
input_model = "runs/train/exp/weights/best.onnx"

# Путь для сохранения оптимизированной модели
output_model = "optimized_model.onnx"

# Оптимизация модели
optimized_model = optimize_model(input_model, output_path=output_model)

print("Модель успешно оптимизирована!")