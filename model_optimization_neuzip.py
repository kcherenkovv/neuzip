import argparse
from neuzip import optimize_model  # Предполагается, что neuzip установлен в окружении

def main():
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Оптимизация модели ONNX с использованием Neuzip.")

    # Добавляем аргументы
    parser.add_argument(
        "input_model",
        type=str,
        help="Путь к исходной модели ONNX"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="optimized_model.onnx",
        help="Путь для сохранения оптимизированной модели (по умолчанию: optimized_model.onnx)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lossless", "lossy"],
        default="lossless",
        help="Режим сжатия: lossless (без потерь) или lossy (с потерями). По умолчанию: lossless"
    )

    # Парсим аргументы
    args = parser.parse_args()

    # Пути к моделям
    input_model = args.input_model
    output_model = args.output_model
    compression_mode = args.mode

    # Оптимизация модели
    try:
        print(f"Запуск оптимизации в режиме: {compression_mode}")
        optimized_model = optimize_model(
            input_model,
            output_path=output_model,
            mode=compression_mode  # <-- Здесь указываем режим сжатия
        )
        print(f"Модель успешно оптимизирована и сохранена в {output_model}!")
    except Exception as e:
        print(f"Произошла ошибка при оптимизации модели: {e}")

if __name__ == "__main__":
    main()
