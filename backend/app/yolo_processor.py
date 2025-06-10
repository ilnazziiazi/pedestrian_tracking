import logging
import os
import cv2
import shutil
import uuid
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from ultralytics import YOLO
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from config import (
    YOLO_MODEL_PATH, DEVICE, PEDESTRIAN_CLASSES,
    YOLO_CONFIDENCE_THRESHOLD, ULTRALYTICS_OUTPUT_DIR
)

logger = logging.getLogger(__name__)


def convert_avi_to_mp4_ffmpeg(avi_path: str) -> str:
    mp4_path = avi_path.replace(".avi", ".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", avi_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            mp4_path
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    logger.info(f"Конвертация AVI → MP4 завершена: {mp4_path}")
    return mp4_path


def get_clustering_features(trajectories: dict) -> np.ndarray:
    features = []

    for track_id, pos in trajectories.items():
        if len(pos) < 2:
            continue

        pos = sorted(pos, key=lambda x: x[0])
        x_start, y_start = pos[0][1:]
        x_end, y_end = pos[-1][1:]
        dx = x_end - x_start
        dy = y_end - y_start
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm == 0:
            continue

        features.append([dx / norm, dy / norm])

    features = np.array(features, dtype=np.float32)

    logger.info(f"Получены фичи кластеризации: {len(features)}")
    return features


def dbscan_clustering(features: np.ndarray) -> np.ndarray:
    # Определяем параметр k ближайших соседей для вычисления дистанций
    k = max(5, int(np.log(len(features))))
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)
    k_distances = np.sort(distances[:, -1])

    # Вычисляем оптимальное eps
    x = np.arange(1, len(k_distances) + 1)
    y = k_distances
    kneedle = KneeLocator(x, y, curve='convex', direction='decreasing')

    if kneedle.knee is None:
        eps = float(np.median(k_distances))
    else:
        eps = y[kneedle.knee]

    # Кластеризуем
    clustering = DBSCAN(eps=eps, min_samples=k, metric='cosine').fit(features)
    labels = clustering.labels_

    logger.info(f"Кластеризация завершена, число кластеров: {len(set(labels)) - (1 if -1 in labels else 0)}, шумов: {list(labels).count(-1)}")
    return labels


class YoloVideoProcessor:
    def __init__(self):
        self.model = None
        self.device = DEVICE
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model.to(self.device)
            logger.info(f"Модель YOLO ({YOLO_MODEL_PATH}) успешно загружена на {self.device}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели YOLO ({YOLO_MODEL_PATH}): {e}", exc_info=True)
            raise

    def process_and_track_video(self, input_video_path: str) -> tuple[str | None, int, dict, dict]:

        experiment_name = f"run_{uuid.uuid4().hex[:12]}"
        output_run_dir = os.path.join(ULTRALYTICS_OUTPUT_DIR, experiment_name)

        processed_video_path = None
        raw_trajectories = defaultdict(list)
        raw_boxes_per_frame = defaultdict(list)

        try:
            logger.info(f"Начало трекинга видео: {input_video_path}")
            logger.info(f"Результаты будут сохранены в директорию внутри: {ULTRALYTICS_OUTPUT_DIR}/{experiment_name}")

            cap = cv2.VideoCapture(input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            results_generator = self.model.track(
                source=input_video_path,
                tracker='bytetrack.yaml',
                save=True,
                save_txt=False,
                save_conf=False,
                classes=PEDESTRIAN_CLASSES,
                conf=YOLO_CONFIDENCE_THRESHOLD,
                project=ULTRALYTICS_OUTPUT_DIR,
                name=experiment_name,
                exist_ok=True,
                verbose=False,
                device=self.device,
                half=True, #FP16 для ускорения
                stream=True
            )

            frame_id = 0
            for result in results_generator:
                if result.boxes is not None and result.boxes.id is not None:
                    # Записываем дату для формирования траекторий движения треков
                    boxes = result.boxes
                    ids = boxes.id.int().cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()
                    xywhn = boxes.xywhn.cpu().tolist()

                    for cls, box, track_id in zip(classes, xywhn, ids):
                        if cls != 4:
                            continue
                        x_center, y_center, w, h = box
                        x_center_px = x_center * width
                        y_center_px = y_center * height
                        raw_trajectories[track_id].append((frame_id, x_center_px, y_center_px))

                        box_w, box_h = w * width, h * height
                        x1 = int(x_center_px - box_w / 2)
                        y1 = int(y_center_px - box_h / 2)
                        x2 = int(x_center_px + box_w / 2)
                        y2 = int(y_center_px + box_h / 2)

                        raw_boxes_per_frame[frame_id].append((track_id, (x1, y1, x2, y2)))

                frame_id += 1
            # Фильтрация по минимальному числу кадров на трек
            valid_track_ids = {
                tid for tid, points in raw_trajectories.items()
                if len(points) >= 20
            }

            trajectories = {tid: raw_trajectories[tid] for tid in valid_track_ids}
            boxes_per_frame = {}

            for frame_id, detections in raw_boxes_per_frame.items():
                filtered = [
                    (tid, box) for tid, box in detections
                    if tid in valid_track_ids
                ]
                if filtered:
                    boxes_per_frame[frame_id] = filtered

            for f_name in os.listdir(output_run_dir):
                if f_name.lower().endswith(".avi"):
                    avi_path = os.path.join(output_run_dir, f_name)
                    processed_video_path = convert_avi_to_mp4_ffmpeg(avi_path)
                    break

            pedestrian_count = len(trajectories)
            logger.info(f"Трекинг для '{input_video_path}' завершен. Уникальных пешеходов: {pedestrian_count}. "
                        f"Сохраненное видео: {processed_video_path}")

            return processed_video_path, pedestrian_count, trajectories, boxes_per_frame

        except Exception as e:
            logger.error(f"Ошибка во время трекинга YOLO для видео {input_video_path}: {e}", exc_info=True)
            if os.path.exists(output_run_dir):
                try:
                    shutil.rmtree(output_run_dir)
                    logger.info(f"Очищена папка эксперимента {output_run_dir} из-за ошибки.")
                except Exception as e_clean:
                    logger.error(f"Ошибка при очистке папки эксперимента {output_run_dir}: {e_clean}")
            return None, 0, {}, {}


    def clusters_visualize(self, input_video_path: str, trajectories: dict,
                           boxes_per_frame: dict, output_path: str, quantile: float = 0.1) -> tuple[str, dict]:
        cap, writer = None, None

        try:
            features = get_clustering_features(trajectories)
            if len(features) == 0:
                logger.warning(f"Пустой список фичей для '{input_video_path}', треки не прошли по длине")
                raise ValueError(f"Пустой список фичей для '{input_video_path}', треки не прошли по длине")

            labels = dbscan_clustering(features)
            if len(set(labels)) == 1 and -1 in labels:
                logger.warning(f"Кластеров для '{input_video_path}' не обнаружено")
                raise ValueError(f"Кластеров для '{input_video_path}' не обнаружено")

            track_ids = list(trajectories.keys())

            df = pd.DataFrame(features, columns=['dx', 'dy'])
            df['track_id'] = track_ids
            df['cluster'] = labels

            cluster_ids = sorted(df['cluster'].unique())
            valid_clusters = [cid for cid in cluster_ids if cid != -1]

            gray_color = (80, 80, 80)
            palette = sns.color_palette("hsv", n_colors=len(valid_clusters))
            palette_bgr = [tuple(int(c * 255) for c in reversed(rgb)) for rgb in palette]

            cluster_colors = dict(zip(valid_clusters, palette_bgr))
            cluster_colors[-1] = gray_color

            track_id_to_cluster = dict(zip(df['track_id'], df['cluster']))

            cluster_track_ids = defaultdict(list)
            for tid, cl in zip(df['track_id'], df['cluster']):
                cluster_track_ids[cl].append(tid)

            cluster_track_counts = {cl: len(set(tids)) for cl, tids in cluster_track_ids.items()}

            cap = cv2.VideoCapture(input_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join(output_path, "clustered_output.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            margin = 200
            cluster_vectors = {}
            for cluster in valid_clusters:
                c_df = df[df['cluster'] == cluster]
                direction = c_df[['dx', 'dy']].mean().values
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                direction /= norm

                all_points = []
                for tid in cluster_track_ids[cluster]:
                    for _, x, y in trajectories.get(tid, []):
                        all_points.append((x, y))

                if len(all_points) < 2:
                    continue

                all_points = np.array(all_points)
                projections = all_points @ direction
                start_pt = all_points[np.argsort(projections)[int(quantile * len(projections))]]
                end_pt = all_points[np.argsort(projections)[int((1 - quantile) * len(projections))]]

                start_pt = np.clip(start_pt, [margin, margin], [width - margin - 1, height - margin - 1])
                end_pt = np.clip(end_pt, [margin, margin], [width - margin - 1, height - margin - 1])

                cluster_vectors[cluster] = (tuple(start_pt.astype(int)), tuple(end_pt.astype(int)))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                for track_id, box in boxes_per_frame.get(frame_idx, []):
                    cluster = track_id_to_cluster.get(track_id, -1)
                    color = cluster_colors.get(cluster, (0, 0, 0))
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{track_id}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                used_label_positions = []
                min_dist = 15

                for cluster, (start_pt, end_pt) in cluster_vectors.items():
                    color = cluster_colors[cluster]
                    cv2.arrowedLine(frame, start_pt, end_pt, color, 2, tipLength=0.02)

                    label_pos = np.array(start_pt, dtype=float)
                    for prev in used_label_positions:
                        while np.linalg.norm(label_pos - prev) < min_dist:
                            label_pos += np.array([0, -15])
                    used_label_positions.append(label_pos)

                    label_pos = tuple(label_pos.astype(int))
                    cv2.putText(frame, f'Cluster {cluster}', label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()
            logger.info(f"Отрисовка кластеров для '{input_video_path}' завершена. Видео сохранено: {out_path}")

            return out_path, cluster_track_counts

        except ValueError as ve:
            logger.warning(f"Ошибка данных при визуализации кластеров для '{input_video_path}': {ve}")
            return "", {}
        except Exception as e:
            logger.error(f"Ошибка при визуализации кластеров для '{input_video_path}': {e}", exc_info=True)
            return "", {}

        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if writer is not None:
                writer.release()


# инициализация, чтобы модель загрузилась один раз при старте FastAPI приложения
try:
    yolo_video_processor = YoloVideoProcessor()
except Exception as e:
    logger.critical(f"Не удалось инициализировать YoloVideoProcessor: {e}", exc_info=True)