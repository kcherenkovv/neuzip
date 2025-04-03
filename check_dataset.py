import os
import cv2
import numpy as np

def validate_yolo_dataset(base_dir):
    """
    Валидирует набор данных YOLO, удаляет непарные файлы и выводит подробный отчет.
    :param base_dir: Базовая директория с папками images/ и labels/
    """
    # Пути к папкам
    train_image_dir = os.path.join(base_dir, 'images', 'train')
    train_label_dir = os.path.join(base_dir, 'labels', 'train')
    test_image_dir = os.path.join(base_dir, 'images', 'test')
    test_label_dir = os.path.join(base_dir, 'labels', 'test')

    # Статистика
    stats = {
        'total': {'images': 0, 'labels': 0, 'objects': 0},
        'removed': {'images': 0, 'labels': 0},
        'class_counts': {},
        'missing_pairs': [],
        'corrupted_files': [],
        'image_sizes': [],
    }

    # Классы (предполагается, что они известны заранее)
    class_names = ['pistol', 'smartphone', 'knife', 'wallet', 'bill', 'card']
    for i in range(len(class_names)):
        stats['class_counts'][i] = 0

    def process_directory(image_dir, label_dir, dataset_type):
        """
        Обрабатывает одну пару папок (например, train или test).
        """
        print(f"\nОбработка {dataset_type}...")

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Ошибка: Директории для {dataset_type} ({image_dir} или {label_dir}) не существуют!")
            return

        # Получаем списки файлов
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        all_files = set([os.path.splitext(f)[0] for f in images + labels])

        for basename in all_files:
            img_ext = next((ext for ext in ['.png', '.jpg', '.jpeg'] if os.path.exists(os.path.join(image_dir, basename + ext))), None)
            img_path = os.path.join(image_dir, basename + (img_ext or ''))
            txt_path = os.path.join(label_dir, basename + '.txt')

            # Проверка существования пары
            if not os.path.exists(img_path) or not os.path.exists(txt_path):
                # Удаление непарных файлов
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        stats['removed']['images'] += 1
                    except OSError as e:
                        print(f"Ошибка при удалении изображения {img_path}: {e}")
                if os.path.exists(txt_path):
                    try:
                        os.remove(txt_path)
                        stats['removed']['labels'] += 1
                    except OSError as e:
                        print(f"Ошибка при удалении аннотации {txt_path}: {e}")

                stats['missing_pairs'].append(basename)
                continue

            # Проверка целостности изображения
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Invalid image file")
                height, width = img.shape[:2]
                stats['image_sizes'].append((width, height))
            except Exception as e:
                stats['corrupted_files'].append(img_path)
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        stats['removed']['images'] += 1
                    except OSError as e:
                        print(f"Ошибка при удалении изображения {img_path}: {e}")
                if os.path.exists(txt_path):
                    try:
                        os.remove(txt_path)
                        stats['removed']['labels'] += 1
                    except OSError as e:
                        print(f"Ошибка при удалении аннотации {txt_path}: {e}")
                continue

            # Проверка аннотаций
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    num_objects = len(lines)
                    stats['total']['objects'] += num_objects

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            raise ValueError("Invalid annotation format")

                        class_id = int(parts[0])
                        if class_id >= len(class_names):
                            raise ValueError(f"Class ID {class_id} превышает количество классов")

                        # Проверка координат
                        coords = list(map(float, parts[1:]))
                        if any(not (0.0 <= x <= 1.0) for x in coords):
                            raise ValueError("Координаты выходят за пределы [0, 1]")

                        stats['class_counts'][class_id] += 1

                stats['total']['images'] += 1
                stats['total']['labels'] += 1

            except Exception as e:
                stats['corrupted_files'].append(txt_path)
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        stats['removed']['images'] += 1
                    except OSError as e:
                        print(f"Ошибка при удалении изображения {img_path}: {e}")
                if os.path.exists(txt_path):
                    try:
                        os.remove(txt_path)
                        stats['removed']['labels'] += 1
                    except OSError as e:
                        print(f"Ошибка при удалении аннотации {txt_path}: {e}")

    # Обработка train и test
    process_directory(train_image_dir, train_label_dir, 'train')
    process_directory(test_image_dir, test_label_dir, 'test')

    # Анализ размеров изображений
    if stats['image_sizes']:
        widths, heights = zip(*stats['image_sizes'])
        min_width, max_width = min(widths), max(widths)
        min_height, max_height = min(heights), max(heights)
        avg_width, avg_height = np.mean(widths), np.mean(heights)
        print("\nImage Size Analysis:")
        print(f"  Min Width: {min_width}, Max Width: {max_width}")
        print(f"  Min Height: {min_height}, Max Height: {max_height}")
        print(f"  Avg Width: {avg_width:.2f}, Avg Height: {avg_height:.2f}")
    else:
        print("\nNo image sizes found.")

    # Анализ балансировки классов
    total_objects = sum(stats['class_counts'].values())
    print("\nClass Distribution:")
    for class_id, count in stats['class_counts'].items():
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"  {class_names[class_id]}: {count} ({percentage:.2f}%)")

    # Вывод итоговой статистики
    print("\nFinal Report:")
    print(f"  Всего изображений: {stats['total']['images']}")
    print(f"  Всего аннотаций: {stats['total']['labels']}")
    print(f"  Всего объектов: {stats['total']['objects']}")
    print(f"  Удалено изображений: {stats['removed']['images']}")
    print(f"  Удалено аннотаций: {stats['removed']['labels']}")

    if stats['corrupted_files']:
        print(f"\nОшибки в файлах: {len(stats['corrupted_files'])} файлов удалено")

# Использование
base_dir = os.path.join(os.getcwd(), "dataset")  # Путь к dataset
validate_yolo_dataset(base_dir)