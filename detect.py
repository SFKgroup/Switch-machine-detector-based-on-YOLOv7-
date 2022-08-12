import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import json
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    counter = {'Error':[0,[]],'WARN':[0,[]],'Warn':[0,[]],'Correct':[0,[]]}
    try:os.mkdir(str(save_dir / 'img'))
    except:pass
    try:os.mkdir(str(save_dir / 'json'))
    except:pass


    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / 'img' / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                out_put = {'wrong':False,'male':[],'female':[],'male_mp':[],'female_mp':[],'error':0,'warn':0,'correct':0}
                #male为[左上坐标,右下坐标];male_mp为[中心坐标,宽,高]

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'

                        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        wide = p2[0]-p1[0]
                        high = p2[1]-p1[1]
                        #print(label,':',high/wide)
                        if label[:-5] == 'male':
                            if float(label[-4:]) < 0.7 or abs(1-high/wide) > 0.2:continue
                        elif label[:-5] == 'female':
                            if float(label[-4:]) < 0.85 or abs(2-high/wide) > 0.35:continue

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        out_put[str(label[:-5])].append([p1, p2,float(label[-4:])])
                        out_put[str(label[:-5])+'_mp'].append([(p1[0]+(p2[0]-p1[0])//2,p1[1]+(p2[1]-p1[1])//2),wide,high,float(label[-4:])])

                ml = 0
                for i in out_put['female_mp']:ml += i[0][1]
                ml = ml//len(out_put['female_mp'])
                long = im0.shape[1]
                if abs(long-ml) >= 500:ml = long
                #print('ml_change:',abs(long-ml))
                
                limit = 0.6

                for i in out_put['male_mp']:
                    in_box = False
                    mid_p = []
                    for k in out_put['female']:
                        if (k[0][0] < i[0][0] and k[1][0] > i[0][0]) and (k[0][1]-20 < i[0][1] and k[1][1]+20 > i[0][1]) and i[2] > limit:
                            in_box = True
                            if out_put['female_mp'][out_put['female'].index(k)][0][1] < ml:mid_p = [out_put['female_mp'][out_put['female'].index(k)][0][0],k[0][1]]
                            else:mid_p = [out_put['female_mp'][out_put['female'].index(k)][0][0],k[1][1]]
                    if in_box:
                        cv2.ellipse(im0,i[0],((i[1])//2,(i[2])//2),0,0,360,(255,255,0),2,lineType=cv2.LINE_AA)
                        per_length = math.sqrt((mid_p[0]-i[0][0])**2+(mid_p[1]-i[0][1])**2)/i[2]
                        #print('length_per:',per_length)
                        if per_length < 2.5:
                            out_put['correct'] += 1
                            cv2.line(im0,mid_p,i[0],(0,255,0),thickness=2,lineType=cv2.LINE_AA)
                        elif per_length >= 2.5 and per_length < 2.9:
                            out_put['warn'] += 1
                            cv2.line(im0,mid_p,i[0],(0,255,255),thickness=2,lineType=cv2.LINE_AA)
                        else:
                            out_put['error'] += 1
                            cv2.line(im0,mid_p,i[0],(0,0,255),thickness=2,lineType=cv2.LINE_AA)
                    else:
                        out_put['male'].pop(out_put['male_mp'].index(i))
                        out_put['male_mp'].remove(i)
                
                for i in out_put['female']:cv2.rectangle(im0, i[0], i[1],(200,200,200),thickness=2, lineType=cv2.LINE_AA)
                
                if out_put['error'] > 0 or (out_put['correct'] == 0 and out_put['warn'] == 0):
                    out_put['wrong'] = True
                    counter['Error'][0] += 1
                    counter['Error'][1].append(save_path)
                    cv2.putText(im0,'Error',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),10)
                    cv2.putText(im0,'Error',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),3)
                elif out_put['warn'] > 2 and out_put['correct'] == 0:
                    out_put['wrong'] = True
                    counter['WARN'][0] += 1
                    counter['WARN'][1].append(save_path)
                    cv2.putText(im0,'WARN',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),10)
                    cv2.putText(im0,'WARN',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,100,255),3)
                elif out_put['warn'] > 1 or out_put['correct'] == 0:
                    counter['Warn'][0] += 1
                    counter['Warn'][1].append(save_path)
                    cv2.putText(im0,'Warn',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),10)
                    cv2.putText(im0,'Warn',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,200,200),3)
                else:
                    counter['Correct'][0] += 1
                    counter['Correct'][1].append(save_path)
                    cv2.putText(im0,'Correct',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),10)
                    cv2.putText(im0,'Correct',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,200,0),3)

                grand = open(str(save_dir / 'json' / p.stem)+'.json','w',encoding='utf-8')
                grand.write(json.dumps(out_put, sort_keys=True, indent=4, separators=(',', ': ')))
                #print(p.stem)
                grand.close()


            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    grand = open(str(save_dir / 'all.json'),'w',encoding='utf-8')
    grand.write(json.dumps(counter, sort_keys=True, indent=4, separators=(',', ': ')))
    grand.close()

    print(f'Done. ({time.time() - t0:.3f}s)')

    for i in counter['Error'][1]:
        img = cv2.imread(i)
        img = cv2.resize(img,None,fx=0.5,fy=0.5)
        cv2.imshow('show',img)
        cv2.waitKey(0)
    for i in counter['WARN'][1]:
        img = cv2.imread(i)
        img = cv2.resize(img,None,fx=0.5,fy=0.5)
        cv2.imshow('show',img)
        cv2.waitKey(0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
