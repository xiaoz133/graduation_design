import cv2
import os

def extract_frames_by_time(video_path, output_dir, interval_sec=0.2):
    """
    从视频中按指定时间间隔提取帧并保存为图片。
    
    :param video_path: 视频文件的路径
    :param output_dir: 保存图片的输出目录
    :param interval_sec: 提取间隔（秒），默认为 0.2 秒
    """
    # 1. 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 2. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}，请检查路径是否正确。")
        return

    # 3. 获取视频的帧率 (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("错误：获取视频帧率失败。")
        return
    
    print(f"视频原始帧率 (FPS): {fps}")

    # 4. 计算每隔多少帧提取一次
    # 比如：如果视频是 30 FPS，0.2秒间隔意味着每 6 帧提取一次 (30 * 0.2 = 6)
    frame_interval = round(fps * interval_sec)
    print(f"每隔 {frame_interval} 帧保存一张图片 (对应 {interval_sec} 秒)")

    frame_count = 0
    saved_count = 0

    # 5. 循环读取视频帧
    while True:
        # ret 是一个布尔值，表示是否成功读取到帧；frame 是当前帧的图像数据
        ret, frame = cap.read()
        
        if not ret:
            break  # 视频读取完毕

        # 如果当前帧的索引是时间间隔的整数倍，则保存该帧
        if frame_count % frame_interval == 0:
            # 格式化文件名，例如：frame_0000.jpg, frame_0001.jpg
            file_name = f"frame_{saved_count:04d}.jpg"
            output_path = os.path.join(output_dir, file_name)
            
            # 保存图片
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    # 6. 释放资源
    cap.release()
    print(f"处理完成！视频总帧数: {frame_count}")
    print(f"共提取并保存了 {saved_count} 张图片到 '{output_dir}' 目录。")

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 视频文件路径
    INPUT_VIDEO = r"C:\Users\94508\Downloads\混凝实验矾花视频\14NTU_16kgoutput_20260410_163251.mp4"
    # 保存图片的文件夹名称
    OUTPUT_FOLDER = r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_folders\normal\14_24.1_16KG"
    # 执行提取，间隔为 0.2 秒
    extract_frames_by_time(INPUT_VIDEO, OUTPUT_FOLDER, interval_sec=0.2)