import cv2
import numpy as np
import os
import json
import time
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
# User-provided base path for videos
VIDEO_CLIPS_BASE_PATH = "C:/Users/hp/Downloads/Hippo/processed_data/clips/" # !UPDATE THIS IF NEEDED!

# Determine the directory where the script is running for relative output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to the script's directory
VIDEO_SOURCE_DIR_ANNOTATION = VIDEO_CLIPS_BASE_PATH
VIDEO_SOURCE_DIR_TESTING = VIDEO_CLIPS_BASE_PATH # Can be changed later for unseen data

ANNOTATIONS_FILE = os.path.join(SCRIPT_DIR, "hippo_annotations_cnn_v3.json")
PATCHES_DIR = os.path.join(SCRIPT_DIR, "hippo_patches_v3/")
CNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "hippo_detector_cnn_v3.h5")

# Specific patch directories
HIPPO1_PATCH_DIR = os.path.join(PATCHES_DIR, "hippo1_large")
HIPPO2_PATCH_DIR = os.path.join(PATCHES_DIR, "hippo2_small")
BACKGROUND_PATCHES_DIR = os.path.join(PATCHES_DIR, "background")

os.makedirs(PATCHES_DIR, exist_ok=True)
os.makedirs(HIPPO1_PATCH_DIR, exist_ok=True)
os.makedirs(HIPPO2_PATCH_DIR, exist_ok=True)
os.makedirs(BACKGROUND_PATCHES_DIR, exist_ok=True)

print(f"--- IMPORTANT SETUP ---")
print(f"1. Annotation Video Source: '{os.path.abspath(VIDEO_SOURCE_DIR_ANNOTATION)}'")
print(f"2. Testing Video Source:    '{os.path.abspath(VIDEO_SOURCE_DIR_TESTING)}'")
print(f"3. Annotation data will be saved/loaded from: '{ANNOTATIONS_FILE}'")
print(f"4. Image patches saved under: '{PATCHES_DIR}'")
print(f"   - Background patches (manual + annotated) go into: '{BACKGROUND_PATCHES_DIR}'")
print(f"5. Trained CNN model saved/loaded as: '{CNN_MODEL_PATH}'")
print(f"6. Before training, populate '{BACKGROUND_PATCHES_DIR}' with diverse background images.")
print(f"-----------------------\n")

# --- CNN & Training Parameters ---
IMG_WIDTH, IMG_HEIGHT = 64, 64; NUM_CLASSES = 3; BACKGROUND_CLASS_IDX = NUM_CLASSES - 1
BATCH_SIZE = 32; EPOCHS = 30

# --- Tracking & Detection Parameters ---
gaussian_blur_kernel_size=(5,5); frame_diff_threshold=18; re_detection_interval_fallback=10
bg_subtractor_type="MOG2"; mog2_history=500; mog2_var_threshold=25; mog2_detect_shadows=False
morph_open_bg_kernel_size=(3,3); morph_close_bg_kernel_size=(17,17)
morph_open_fd_kernel_size=(3,3); morph_close_fd_kernel_size=(15,15)

# --- Hippo Profiles & Behavior ---
HIPPO_PROFILES = {
    1: {"name":"Hippo 1 (L)","min_area":700,"max_area":40000,"min_solidity":0.60,"min_aspect":0.4,"max_aspect":3.0,
        "tracker":None,"bbox":None,"id":1,"patch_dir":HIPPO1_PATCH_DIR,"last_seen_frame":0,"class_idx":0,
        "prev_bbox":None,"prev_center":None,"current_behavior":"unknown","current_emotion":"Neutral"},
    2: {"name":"Hippo 2 (S)","min_area":100,"max_area":6000,"min_solidity":0.55,"min_aspect":0.3,"max_aspect":2.8,
        "tracker":None,"bbox":None,"id":2,"patch_dir":HIPPO2_PATCH_DIR,"last_seen_frame":0,"class_idx":1,
        "prev_bbox":None,"prev_center":None,"current_behavior":"unknown","current_emotion":"Neutral"}
}
ANNOTATED_BACKGROUND_ID = 0; ANNOTATED_BACKGROUND_NAME = "Annotated BG"

BEHAVIOR_CLASSES = ["resting","feeding","walking","swimming","interacting_with_object","social_interaction_positive",
                    "social_interaction_negative","vigilant","wallowing","pacing_stereotypy","playing","sleeping",
                    "out_of_frame","other_active"]
EMOTION_CLASSES = ["Happy","Calm","Curious","Playful","Alert","Neutral"]

DISTRACTOR_HUMAN_MIN_ASPECT=0.2;DISTRACTOR_HUMAN_MAX_ASPECT=0.8;DISTRACTOR_HUMAN_MIN_SOLIDITY=0.70;DISTRACTOR_HUMAN_MIN_AREA_FILTER=500
DISTRACTOR_SHADOW_MAX_SOLIDITY=0.60;DISTRACTOR_SHADOW_MIN_AREA_FILTER=200

max_hippos_to_track=len(HIPPO_PROFILES);tracker_type_cv="CSRT";cnn_confidence_threshold=0.75

# --- Global variables ---
in_annotation_mode=False;annotating_id_pending=None;current_roi_points=[]
temp_frame_for_annotation=None;current_video_path_global="";current_frame_num_global=0
paused_playback=False;loaded_cnn_model=None;annotations_data=[]

if os.path.exists(ANNOTATIONS_FILE):
    try:
        with open(ANNOTATIONS_FILE,'r')as f:annotations_data=json.load(f)
        print(f"Loaded {len(annotations_data)} annotations from {ANNOTATIONS_FILE}.")
    except Exception as e: print(f"Warn: Load annotations {ANNOTATIONS_FILE} failed: {e}")
else: print(f"Annotation file {ANNOTATIONS_FILE} not found. Starting fresh.")

def save_annotations_to_file():
    try:
        with open(ANNOTATIONS_FILE,'w')as f:json.dump(annotations_data,f,indent=4)
        print(f"Annotations saved ({len(annotations_data)}) to {ANNOTATIONS_FILE}.")
    except Exception as e:print(f"ERR save annotations:{e}")

def on_mouse_draw_roi(event,x,y,flags,param):
    global current_roi_points,in_annotation_mode,temp_frame_for_annotation,annotating_id_pending,current_video_path_global,current_frame_num_global,HIPPO_PROFILES,annotations_data
    if not in_annotation_mode or temp_frame_for_annotation is None or annotating_id_pending is None:return
    is_bg_anno=(annotating_id_pending==ANNOTATED_BACKGROUND_ID)
    name=ANNOTATED_BACKGROUND_NAME if is_bg_anno else HIPPO_PROFILES[annotating_id_pending]["name"]
    patch_dir=BACKGROUND_PATCHES_DIR if is_bg_anno else HIPPO_PROFILES[annotating_id_pending]["patch_dir"]
    class_idx=BACKGROUND_CLASS_IDX if is_bg_anno else HIPPO_PROFILES[annotating_id_pending]["class_idx"]
    draw_clr=(0,165,255)if is_bg_anno else(0,255,255); disp_frame=temp_frame_for_annotation.copy()
    if event==cv2.EVENT_LBUTTONDOWN:current_roi_points=[(x,y)]
    elif event==cv2.EVENT_MOUSEMOVE and len(current_roi_points)==1:
        cv2.rectangle(disp_frame,current_roi_points[0],(x,y),draw_clr,2);cv2.imshow('Hippo Annotation/Test',disp_frame)
    elif event==cv2.EVENT_LBUTTONUP:
        current_roi_points.append((x,y))
        if len(current_roi_points)==2:
            x1,y1=current_roi_points[0];x2,y2=current_roi_points[1];bbox=(min(x1,x2),min(y1,y2),abs(x1-x2),abs(y1-y2))
            if bbox[2]>5 and bbox[3]>5:
                ts=time.strftime("%Y%m%d_%H%M%S");vid_base=os.path.basename(current_video_path_global)
                s_vid_name="".join(c if c.isalnum()else"_"for c in vid_base.split('.')[0])
                fprefix="bg_annot_"if is_bg_anno else f"hippo{annotating_id_pending}_"
                p_fname=f"{fprefix}{s_vid_name}_f{current_frame_num_global}_{ts}.png";p_path=os.path.join(patch_dir,p_fname)
                patch_img=temp_frame_for_annotation[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
                if patch_img.size==0:print("Err:Empty patch!");current_roi_points=[];return
                cv2.imwrite(p_path,patch_img);annotations_data.append({"video_path":current_video_path_global,
                "frame_number":current_frame_num_global,"label_id":annotating_id_pending,"class_idx":class_idx,
                "bbox":list(bbox),"patch_path":p_path})
                print(f"Annotated {name},bbox:{bbox}.Patch:{p_path}");save_annotations_to_file()
                if not is_bg_anno:
                    prof=HIPPO_PROFILES[annotating_id_pending]
                    try:
                        if prof["tracker"] is not None:prof["tracker"]=None
                        trk=create_cv_tracker(tracker_type_cv);trk.init(temp_frame_for_annotation,bbox);
                        prof.update({"tracker":trk,"bbox":bbox,"last_seen_frame":current_frame_num_global})
                    except Exception as e:print(f"Err manual trk init:{e}")
                annotating_id_pending=None;current_roi_points=[]
                box_color=(100,100,100)if is_bg_anno else tracker_colors[HIPPO_PROFILES[annotating_id_pending if annotating_id_pending is not None and annotating_id_pending!=ANNOTATED_BACKGROUND_ID else 1]["id"]-1]
                if not is_bg_anno and annotating_id_pending is not None:pass
                else:cv2.rectangle(disp_frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),box_color,2)
                cv2.imshow('Hippo Annotation/Test',disp_frame)
            else:current_roi_points=[];cv2.imshow('Hippo Annotation/Test',temp_frame_for_annotation.copy());print("ROI small.")

def create_cv_tracker(tracker_type_str_local):
    if tracker_type_str_local=='CSRT':return cv2.TrackerCSRT_create()
    else:return cv2.TrackerCSRT_create()

def run_motion_detector(frame_local,prev_gray_local,back_sub_local):
    cand_props=[];proc_f=cv2.GaussianBlur(frame_local,gaussian_blur_kernel_size,0);gray_f=cv2.cvtColor(proc_f,cv2.COLOR_BGR2GRAY)
    fg_raw=back_sub_local.apply(proc_f,learningRate=0.001)
    fg_bg=cv2.morphologyEx(cv2.morphologyEx(fg_raw,cv2.MORPH_OPEN,open_kernel_bg),cv2.MORPH_CLOSE,close_kernel_bg)
    cnts_bg,_=cv2.findContours(fg_bg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);all_cnts=[{'c':c,'s':'bg'}for c in cnts_bg]
    if prev_gray_local is not None:
        f_diff=cv2.absdiff(prev_gray_local,gray_f);_,th_diff=cv2.threshold(f_diff,frame_diff_threshold,255,cv2.THRESH_BINARY)
        dil_diff=cv2.dilate(th_diff,open_kernel_fd,iterations=2);fg_fd=cv2.morphologyEx(dil_diff,cv2.MORPH_CLOSE,close_kernel_fd)
        cnts_fd,_=cv2.findContours(fg_fd,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);[all_cnts.append({'c':c,'s':'fd'})for c in cnts_fd]
    for item in all_cnts:
        c=item['c'];area=cv2.contourArea(c);x,y,w,h=cv2.boundingRect(c)
        if area<50:continue
        ar=float(w)/(h+1e-6);hull=cv2.convexHull(c);sol=float(area)/(cv2.contourArea(hull)+1e-6)if cv2.contourArea(hull)>0 else 0
        is_h=(DISTRACTOR_HUMAN_MIN_ASPECT<ar<DISTRACTOR_HUMAN_MAX_ASPECT and sol>DISTRACTOR_HUMAN_MIN_SOLIDITY and area>DISTRACTOR_HUMAN_MIN_AREA_FILTER)
        is_s=(sol<DISTRACTOR_SHADOW_MAX_SOLIDITY and area>DISTRACTOR_SHADOW_MIN_AREA_FILTER)
        if is_h or is_s:continue
        cand_props.append({'contour':c,'bbox':(x,y,w,h),'area':area,'solidity':sol,'aspect_ratio':ar,'source':item['s']})
    return cand_props,gray_f,fg_raw,fg_bg

def create_lightweight_cnn(input_shape=(IMG_HEIGHT,IMG_WIDTH,3),num_classes=NUM_CLASSES):
    model=Sequential([Conv2D(32,(3,3),activation='relu',input_shape=input_shape,padding='same'),BatchNormalization(),MaxPooling2D((2,2)),
                      Conv2D(64,(3,3),activation='relu',padding='same'),BatchNormalization(),MaxPooling2D((2,2)),
                      Conv2D(128,(3,3),activation='relu',padding='same'),BatchNormalization(),MaxPooling2D((2,2)),
                      Flatten(),Dense(128,activation='relu'),BatchNormalization(),Dropout(0.5),Dense(num_classes,activation='softmax')])
    model.compile(optimizer=Adam(learning_rate=0.0005),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def train_cnn_model_mode():
    global loaded_cnn_model;print("\n--- CNN Training ---");images=[];labels=[];loaded_paths=set()
    if not annotations_data:print("No JSON annotations. Annotate first.");
    for ann in annotations_data:
        if not os.path.exists(ann["patch_path"]):print(f"Warn:Patch missing {ann['patch_path']}");continue
        img=cv2.imread(ann["patch_path"])
        if img is not None:img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT));images.append(img);labels.append(ann["class_idx"]);loaded_paths.add(os.path.normpath(ann["patch_path"]))
    bg_gen_count=0
    if os.path.exists(BACKGROUND_PATCHES_DIR):
        for img_n in os.listdir(BACKGROUND_PATCHES_DIR):
            if img_n.lower().endswith(('.png','.jpg','.jpeg')):
                img_p=os.path.join(BACKGROUND_PATCHES_DIR,img_n)
                if os.path.normpath(img_p)in loaded_paths:continue
                img=cv2.imread(img_p)
                if img is not None:img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT));images.append(img);labels.append(BACKGROUND_CLASS_IDX);bg_gen_count+=1
    print(f"Loaded {bg_gen_count} general BG patches.")
    n_h1=labels.count(HIPPO_PROFILES[1]["class_idx"]);n_h2=labels.count(HIPPO_PROFILES[2]["class_idx"]);n_bg=labels.count(BACKGROUND_CLASS_IDX)
    print(f"Total Patches: H1={n_h1}, H2={n_h2}, BG={n_bg}")
    if not images or len(set(labels))<NUM_CLASSES or n_bg==0:print(f"Insufficient data. Need H1,H2,>0 BG. Unique classes:{len(set(labels))}.");return
    if len(images)<BATCH_SIZE:print(f"Warn:Samples({len(images)})<batch({BATCH_SIZE}).");return
    imgs_np=np.array(images,dtype="float32")/255.0;lbls_np=to_categorical(np.array(labels),num_classes=NUM_CLASSES)
    trX,tsX,trY,tsY=train_test_split(imgs_np,lbls_np,test_size=0.20,stratify=lbls_np if len(set(labels))>1 else None,random_state=42)
    print(f"Train:{trX.shape}, Test:{tsX.shape}")
    aug=ImageDataGenerator(rotation_range=15,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,fill_mode="nearest")
    model=create_lightweight_cnn();print("Training...");
    hist=model.fit(aug.flow(trX,trY,batch_size=BATCH_SIZE),validation_data=(tsX,tsY),epochs=EPOCHS,verbose=1,
                   workers=max(1,os.cpu_count()//2),use_multiprocessing=True if os.name!='nt'else False)
    model.save(CNN_MODEL_PATH);print(f"Model saved:{CNN_MODEL_PATH}");loaded_cnn_model=model

def run_cnn_detector(frame_local,motion_props_local):
    if loaded_cnn_model is None:return{hid:[]for hid in HIPPO_PROFILES}
    dets={hid:[]for hid in HIPPO_PROFILES};patches_cls=[];orig_infos=[]
    for prop in motion_props_local:
        x,y,w,h=prop['bbox']
        if w<10 or h<10:continue
        patch=frame_local[y:y+h,x:x+w];
        if patch.size==0:continue
        r_patch=cv2.resize(patch,(IMG_WIDTH,IMG_HEIGHT));n_patch=r_patch.astype("float32")/255.0
        patches_cls.append(n_patch);orig_infos.append(prop)
    if not patches_cls:return dets
    patch_arr=np.array(patches_cls)
    if patch_arr.ndim==3 and len(patches_cls)==1:patch_arr=np.expand_dims(patch_arr,axis=0)
    elif patch_arr.ndim!=4 and len(patches_cls)>1:print(f"Warn:patch_arr ndim {patch_arr.ndim}");return dets
    if patch_arr.shape[0]==0:return dets
    preds=loaded_cnn_model.predict(patch_arr)
    for i,scores in enumerate(preds):
        orig_info=orig_infos[i];pred_idx=np.argmax(scores);conf=scores[pred_idx]
        if conf>=cnn_confidence_threshold:
            target_hid=None
            for hid_map,prof_map in HIPPO_PROFILES.items():
                if prof_map["class_idx"]==pred_idx:target_hid=hid_map;break
            if target_hid is not None:
                prof=HIPPO_PROFILES[target_hid]
                area=orig_info['area'];aspect=orig_info['aspect_ratio'];solidity=orig_info['solidity']
                if(prof["min_area"]<area<prof["max_area"]and prof["min_aspect"]<aspect<prof["max_aspect"]and solidity>prof["min_solidity"]):
                    dets[target_hid].append({'bbox':orig_info['bbox'],'confidence':float(conf),'source':'cnn_'+orig_info['source']})
    return dets

def infer_behavior_and_emotion(hippo1_profile, hippo2_profile, frame_area): # (Same as previous)
    # ... (Your infer_behavior_and_emotion function from previous version)
    behaviors={1:"out_of_frame",2:"out_of_frame"};emotions={1:"Neutral",2:"Neutral"};velocities={1:0,2:0}
    for hid,profile in[(1,hippo1_profile),(2,hippo2_profile)]:
        if profile["bbox"]and profile["prev_bbox"]:
            c_cx=profile["bbox"][0]+profile["bbox"][2]/2;c_cy=profile["bbox"][1]+profile["bbox"][3]/2
            p_cx=profile["prev_bbox"][0]+profile["prev_bbox"][2]/2;p_cy=profile["prev_bbox"][1]+profile["prev_bbox"][3]/2
            dist=np.sqrt((c_cx-p_cx)**2+(c_cy-p_cy)**2);velocities[hid]=dist/((profile["bbox"][2]+profile["bbox"][3])/2+1e-6)
        elif profile["bbox"]:velocities[hid]=0.1
    social_interaction="none"
    if hippo1_profile["bbox"]and hippo2_profile["bbox"]:
        c1x=hippo1_profile["bbox"][0]+hippo1_profile["bbox"][2]/2;c1y=hippo1_profile["bbox"][1]+hippo1_profile["bbox"][3]/2
        c2x=hippo2_profile["bbox"][0]+hippo2_profile["bbox"][2]/2;c2y=hippo2_profile["bbox"][1]+hippo2_profile["bbox"][3]/2
        dist_h=np.sqrt((c1x-c2x)**2+(c1y-c2y)**2)
        s1=(hippo1_profile["bbox"][2]+hippo1_profile["bbox"][3])/2 if hippo1_profile["bbox"] else 1
        s2=(hippo2_profile["bbox"][2]+hippo2_profile["bbox"][3])/2 if hippo2_profile["bbox"] else 1
        if dist_h<(s1+s2)*0.75:social_interaction="close_proximity"
    for hid,profile in[(1,hippo1_profile),(2,hippo2_profile)]:
        if profile["bbox"]is None:behaviors[hid]="out_of_frame";emotions[hid]="Neutral";continue
        vel=velocities[hid];area_b=profile["bbox"][2]*profile["bbox"][3]
        if vel<0.02:behaviors[hid]="sleeping"if area_b<(0.01*frame_area)else "resting";emotions[hid]="Calm"
        elif vel<0.15:behaviors[hid]="walking";emotions[hid]="Neutral"
        elif vel<0.5:behaviors[hid]="walking";emotions[hid]="Alert"
        else:behaviors[hid]="playing";emotions[hid]="Playful"
    if social_interaction=="close_proximity":
        if velocities[1]<0.15 and velocities[2]<0.15:behaviors[1]=behaviors[2]="social_interaction_positive";emotions[1]=emotions[2]="Calm"
        elif(velocities[1]>0.4 and velocities[2]<0.2)or(velocities[2]>0.4 and velocities[1]<0.2):behaviors[1]=behaviors[2]="playing";emotions[1]=emotions[2]="Playful"
    hippo1_profile["current_behavior"]=behaviors[1];hippo1_profile["current_emotion"]=emotions[1]
    hippo2_profile["current_behavior"]=behaviors[2];hippo2_profile["current_emotion"]=emotions[2]
    return behaviors,emotions


def annotate_videos_mode():
    global current_video_path_global,current_frame_num_global,paused_playback,in_annotation_mode, annotating_id_pending
    global temp_frame_for_annotation,prev_gray_frame_for_detector_diff,HIPPO_PROFILES,tracker_colors,user_wants_to_quit_program_globally

    cv2.namedWindow('Hippo Annotation/Test')
    video_files=[os.path.join(VIDEO_SOURCE_DIR_ANNOTATION,f)for f in os.listdir(VIDEO_SOURCE_DIR_ANNOTATION)if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
    if not video_files:print(f"No videos in ANNOTATION dir:'{VIDEO_SOURCE_DIR_ANNOTATION}'.");return
    user_wants_to_quit_program_globally=False
    for video_idx,vid_path in enumerate(video_files):
        if user_wants_to_quit_program_globally:break
        current_video_path_global=vid_path;cap=cv2.VideoCapture(vid_path)
        if not cap.isOpened():print(f"Error opening:{vid_path}");continue
        print(f"\n--- Annotating Vid {video_idx+1}/{len(video_files)}:{os.path.basename(vid_path)} ---")
        print("Ctrl:[a]Anno|[p]Pause|[n]NextVid|[q]MainMenu || Anno:[0]BG(PRIORITY)|[1]H1|[2]H2|[ESC]CancelSel|[a]ExitAnno")
        for hid_r in HIPPO_PROFILES:HIPPO_PROFILES[hid_r].update({"tracker":None,"bbox":None,"last_seen_frame":0})
        current_frame_num_global=0;paused_playback=False;in_annotation_mode=False;annotating_id_pending=None
        prev_gray_frame_for_detector_diff=None
        backSub_instance=cv2.createBackgroundSubtractorMOG2(history=mog2_history,varThreshold=mog2_var_threshold,detectShadows=mog2_detect_shadows)
        latest_frame_data=None;skip_to_next_video_flag=False;raw_bg_mask_disp=None;clean_bg_mask_disp=None
        while True:
            if not paused_playback or(in_annotation_mode and annotating_id_pending is None):
                ret,live_frame=cap.read()
                if not ret:break
                current_frame_num_global+=1;latest_frame_data=live_frame.copy()
            elif paused_playback and latest_frame_data is None and ret:latest_frame_data=live_frame.copy()
            display_frame=latest_frame_data.copy()if latest_frame_data is not None else np.zeros((480,640,3),dtype=np.uint8)
            if in_annotation_mode:
                text_anno=f"ANNOTATE BG/HIPPO (F{current_frame_num_global}): "
                if annotating_id_pending is not None:
                    name=ANNOTATED_BACKGROUND_NAME if annotating_id_pending==ANNOTATED_BACKGROUND_ID else HIPPO_PROFILES[annotating_id_pending]['name']
                    text_anno+=f"Draw {name}.[ESC]cancel"
                else:text_anno+="[0]BG(PRIORITY)|[1]H1|[2]H2|[a]ExitAnno"
                cv2.putText(display_frame,text_anno,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            elif paused_playback:cv2.putText(display_frame,f"PAUSED (F{current_frame_num_global}).[p]resume,[a]anno",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            if not(in_annotation_mode and annotating_id_pending is not None and len(current_roi_points)==1):
                _,cur_gray,raw_bg_mask_disp,clean_bg_mask_disp=run_motion_detector(display_frame,prev_gray_frame_for_detector_diff,backSub_instance)
                prev_gray_frame_for_detector_diff=cur_gray
                for hid_upd,prof_upd in HIPPO_PROFILES.items():
                    if prof_upd["tracker"]is not None:
                        succ,bbox_u=prof_upd["tracker"].update(display_frame)
                        if succ:x,y,w,h=[int(v)for v in bbox_u];prof_upd["bbox"]=(x,y,w,h);cv2.rectangle(display_frame,(x,y),(x+w,y+h),tracker_colors[hid_upd-1],2);cv2.putText(display_frame,prof_upd["name"],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,tracker_colors[hid_upd-1],2)
                        else:prof_upd["tracker"]=None
                if current_frame_num_global%30<2 and raw_bg_mask_disp is not None and clean_bg_mask_disp is not None:
                    cv2.imshow('Raw BG Mask',raw_bg_mask_disp);cv2.imshow('Cleaned BG Mask',clean_bg_mask_disp)
            cv2.imshow('Hippo Annotation/Test',display_frame)
            wait_key_val=0 if paused_playback or(in_annotation_mode and annotating_id_pending)else 1
            key=cv2.waitKey(wait_key_val)
            if key!=-1:key&=0xFF
            if key==ord('q'):user_wants_to_quit_program_globally=True;break
            elif key==ord('n'):skip_to_next_video_flag=True;break
            elif key==ord('p'):
                paused_playback=not paused_playback
                if paused_playback and in_annotation_mode:annotating_id_pending=None
                elif not paused_playback and in_annotation_mode:in_annotation_mode=False;annotating_id_pending=None;cv2.setMouseCallback('Hippo Annotation/Test',lambda*args:None)
                print("Paused"if paused_playback else"Resumed")
            elif key==ord('a'):
                in_annotation_mode=not in_annotation_mode
                if in_annotation_mode:
                    paused_playback=True;annotating_id_pending=None;current_roi_points=[]
                    if latest_frame_data is not None:temp_frame_for_annotation=latest_frame_data.copy()
                    else:in_annotation_mode=False;paused_playback=False;continue
                    cv2.setMouseCallback('Hippo Annotation/Test',on_mouse_draw_roi)
                    print(f"AnnoMode ON(F{current_frame_num_global}).Select ID [0]BG(PRIORITY)|[1]H1|[2]H2.")
                else:paused_playback=False;annotating_id_pending=None;cv2.setMouseCallback('Hippo Annotation/Test',lambda*args:None);print("AnnoMode OFF.Resuming.")
            elif in_annotation_mode and annotating_id_pending is None:
                if key==ord('1'):annotating_id_pending=1
                elif key==ord('2'):annotating_id_pending=2
                elif key==ord('0'):annotating_id_pending=ANNOTATED_BACKGROUND_ID
                if annotating_id_pending is not None:
                    current_roi_points=[]
                    name_to_draw=ANNOTATED_BACKGROUND_NAME if annotating_id_pending==ANNOTATED_BACKGROUND_ID else HIPPO_PROFILES[annotating_id_pending]['name']
                    print(f"Selected {name_to_draw}.Draw box.")
                    if temp_frame_for_annotation is None and latest_frame_data is not None:temp_frame_for_annotation=latest_frame_data.copy()
            elif in_annotation_mode and annotating_id_pending is not None and key==27: # ESC
                annotating_id_pending=None;current_roi_points=[]
                print("ID selection cancelled.");
                if temp_frame_for_annotation is not None:cv2.imshow('Hippo Annotation/Test',temp_frame_for_annotation.copy())
        cap.release()
        if skip_to_next_video_flag or user_wants_to_quit_program_globally:continue
    print("Exiting Annotation Mode.")

def test_cnn_model_mode(): # (Modified to use HIPPO_PROFILES for behavior state)
    global loaded_cnn_model,current_frame_num_global,prev_gray_frame_for_detector_diff,user_wants_to_quit_program_globally, HIPPO_PROFILES
    if loaded_cnn_model is None:
        if os.path.exists(CNN_MODEL_PATH):
            try:loaded_cnn_model=load_model(CNN_MODEL_PATH);print(f"Loaded CNN for testing:{CNN_MODEL_PATH}")
            except Exception as e:print(f"Err load CNN {CNN_MODEL_PATH}:{e}.Train first.");return
        else:print("No trained CNN.Train first ('2' in main menu).");return
    cv2.namedWindow('Hippo Annotation/Test')
    video_files=[os.path.join(VIDEO_SOURCE_DIR_TESTING,f)for f in os.listdir(VIDEO_SOURCE_DIR_TESTING)if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
    if not video_files:print(f"No videos in TESTING dir:'{VIDEO_SOURCE_DIR_TESTING}'.");return
    user_wants_to_quit_program_globally=False
    for video_idx,vid_path in enumerate(video_files):
        if user_wants_to_quit_program_globally:break
        cap=cv2.VideoCapture(vid_path);
        if not cap.isOpened():print(f"Err open test vid:{vid_path}");continue
        print(f"\n--- Testing Vid {video_idx+1}/{len(video_files)}:{os.path.basename(vid_path)} ---")
        print("Displaying CNN detections & behaviors. Press [q] for next video/main menu.")
        current_frame_num_global=0;prev_gray_frame_for_detector_diff=None
        backSub_instance_test=cv2.createBackgroundSubtractorMOG2(history=mog2_history,varThreshold=mog2_var_threshold,detectShadows=mog2_detect_shadows)
        for hid_reset_beh in HIPPO_PROFILES:HIPPO_PROFILES[hid_reset_beh].update({"prev_bbox":None,"bbox":None}) # Reset bboxes for behavior
        frame_pixel_area=0
        while True:
            ret,frame=cap.read();
            if not ret:break
            current_frame_num_global+=1;display_frame=frame.copy()
            if frame_pixel_area==0:frame_pixel_area=frame.shape[0]*frame.shape[1]
            for hid_store_prev in HIPPO_PROFILES:HIPPO_PROFILES[hid_store_prev]["prev_bbox"]=HIPPO_PROFILES[hid_store_prev]["bbox"];HIPPO_PROFILES[hid_store_prev]["bbox"]=None
            motion_props,cur_gray,_,_=run_motion_detector(display_frame,prev_gray_frame_for_detector_diff,backSub_instance_test)
            prev_gray_frame_for_detector_diff=cur_gray
            cnn_detections=run_cnn_detector(display_frame,motion_props)
            detected_this_frame={1:None,2:None}
            for hippo_id_cnn,detections_list in cnn_detections.items():
                if detections_list:
                    best_det=max(detections_list,key=lambda d:d['confidence'])
                    if best_det['confidence']>=cnn_confidence_threshold:HIPPO_PROFILES[hippo_id_cnn]["bbox"]=best_det['bbox'];detected_this_frame[hippo_id_cnn]=best_det
            inferred_behaviors,inferred_emotions=infer_behavior_and_emotion(HIPPO_PROFILES[1],HIPPO_PROFILES[2],frame_pixel_area)
            for hippo_id_disp in[1,2]:
                prof_disp=HIPPO_PROFILES[hippo_id_disp]
                if prof_disp["bbox"]is not None:
                    x,y,w,h=prof_disp["bbox"];class_name=prof_disp["name"];conf_text=""
                    if detected_this_frame[hippo_id_disp]:conf_text=f"(CNN:{detected_this_frame[hippo_id_disp]['confidence']:.2f})"
                    lbl1=f"{class_name}{conf_text}";lbl2=f"Beh:{prof_disp['current_behavior']}";lbl3=f"Emo:{prof_disp['current_emotion']}"
                    cv2.rectangle(display_frame,(x,y),(x+w,y+h),tracker_colors[hippo_id_disp-1],2)
                    cv2.putText(display_frame,lbl1,(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,tracker_colors[hippo_id_disp-1],1)
                    cv2.putText(display_frame,lbl2,(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    cv2.putText(display_frame,lbl3,(x,y-0),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.imshow('Hippo Annotation/Test',display_frame)
            key=cv2.waitKey(50)&0xFF
            if key==ord('q'):user_wants_to_quit_program_globally=True;break
        cap.release()
        if user_wants_to_quit_program_globally:break
    print("Exiting Testing Mode.")

# --- Main Application Menu & Loop ---
if __name__ == "__main__":
    if VIDEO_SOURCE_DIR_ANNOTATION == "path/to/your/video_clips_for_annotation/" or \
       not os.path.isdir(VIDEO_SOURCE_DIR_ANNOTATION):
        print("CRITICAL ERROR: 'VIDEO_SOURCE_DIR_ANNOTATION' not set correctly or path not found."); exit()
    if VIDEO_SOURCE_DIR_TESTING == "path/to/your/video_clips_for_testing/" or \
       not os.path.isdir(VIDEO_SOURCE_DIR_TESTING):
        print("WARNING: 'VIDEO_SOURCE_DIR_TESTING' not set correctly or path not found. Testing mode may fail.");

    tracker_colors=[(0,255,0),(0,0,255),(100,100,100)]
    open_kernel_bg=cv2.getStructuringElement(cv2.MORPH_RECT,morph_open_bg_kernel_size)
    close_kernel_bg=cv2.getStructuringElement(cv2.MORPH_RECT,morph_close_bg_kernel_size)
    open_kernel_fd=cv2.getStructuringElement(cv2.MORPH_RECT,morph_open_fd_kernel_size)
    close_kernel_fd=cv2.getStructuringElement(cv2.MORPH_RECT,morph_close_fd_kernel_size)

    if os.path.exists(CNN_MODEL_PATH)and loaded_cnn_model is None:
        try:loaded_cnn_model=load_model(CNN_MODEL_PATH);print(f"Loaded CNN model at start:{CNN_MODEL_PATH}")
        except Exception as e:print(f"Notice:Err load CNN at start({CNN_MODEL_PATH}):{e}.Train using [2].")
    while True:
        print("\n--- Hippo Behavior & Tracking Application ---")
        print("Choose an action:")
        print("  [1] Annotate Videos (Focus: BACKGROUND [0], Hippos [1]/[2])")
        print("  [2] Train CNN Model (Requires Annotations & BG Patches)")
        print("  [3] Test Trained CNN Model (View Detections & Behaviors)")
        print("  [q] Quit Application")
        choice=input("Enter choice:").strip().lower()
        if choice=='1':annotate_videos_mode()
        elif choice=='2':train_cnn_model_mode()
        elif choice=='3':test_cnn_model_mode()
        elif choice=='q':print("Exiting.");break
        else:print("Invalid choice.")
    cv2.destroyAllWindows()
    print("Application finished.")