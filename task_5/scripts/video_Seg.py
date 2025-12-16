from ultralytics import YOLO
import cv2
from pathlib import Path
from natsort import natsorted


model = YOLO('/home/sashreekkumar/Documents/IIITH-AI-ML-Internship/polyp_seg/yolo11n_run5/weights/best.pt')

images_folder = '/home/sashreekkumar/Documents/IIITH-AI-ML-Internship/colonoscopy-sequences/positive_cropped/seq2/images'
output_video = 'colonoscopy_polyp_detection.mp4'

fps = 25
conf_threshold = 0.25


def process_images_to_video(images_folder, model, output_video, fps=25, conf=0.25):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_folder).glob(f'*{ext}'))
        image_files.extend(Path(images_folder).glob(f'*{ext.upper()}'))
    
    image_files = natsorted([str(f) for f in image_files])
    
    if len(image_files) == 0:
        print(f"No images found in {images_folder}")
        return
    
    print(f"Found {len(image_files)} images")
    
    first_img = cv2.imread(image_files[0])
    height, width = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    total_polyps_detected = 0
    frames_with_polyps = []
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"Output video: {output_video} at {fps} FPS")
    print("-" * 50)
    
    for idx, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        results = model(frame, conf=conf, iou=0.7, verbose=False)
        annotated_frame = results[0].plot()
        
        num_detections = len(results[0].boxes)
        if num_detections > 0:
            total_polyps_detected += num_detections
            frames_with_polyps.append(idx)
            
            cv2.putText(annotated_frame, f"POLYP DETECTED ({num_detections})", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 3)
        
        cv2.putText(annotated_frame, f"Frame: {idx+1}/{len(image_files)}", 
                   (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(image_files)} frames...")
    
    out.release()
    
    print("\n" + "="*50)
    print("Processing Complete")
    print("="*50)
    print(f"Total frames processed: {len(image_files)}")
    print(f"Frames with polyps: {len(frames_with_polyps)}")
    print(f"Total polyp detections: {total_polyps_detected}")
    print(f"Output video: {output_video}")
    

if __name__ == "__main__":
    frames_with_polyps = process_images_to_video(
        images_folder=images_folder,
        model=model,
        output_video=output_video,
        fps=fps,
        conf=conf_threshold
    )
    
    print(output_video)
