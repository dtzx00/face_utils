#!/usr/bin/env python
# coding: utf-8

import cv2, os, operator, math, time, pickle, itertools
import base64, requests, collections, random
import pandas as pd; import numpy as np
from matplotlib import pyplot as plt
from dotenv import dotenv_values

# TensorFlow / Keras are imported lazily inside load_custom_vgg() so that the
# rest of the package works without a heavy deep-learning install.

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split





def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False

def Get_Flattened_Dict(d, parent_key='', sep='_'):
    '''Utility function that flattens nested dictionary
    from Face++ and returns a flattened dictionary
    '''
    items = []
    for k, v in d.items():
        new_key = parent_key+sep+k if parent_key else k
        if isinstance(v,collections.abc.MutableMapping):
            items.extend(Get_Flattened_Dict(v,new_key,sep=sep).items())
        else:
            items.append((new_key,v))
    return dict(items)

def Get_Euclidean_Distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    return np.sqrt(euclidean_distance)

def load_api_keys(location='.env'):
    '''
    Go to https://www.faceplusplus.com and obtain API key + secret.
    Put your keys and secrets into a hiden file like this:
    
    export FREE_KEY = 'X6x5x7x8x6c6c7x8x76c6c8x8c7'
    export FREE_SECRET = 'Xxx78x7x7x8x78x7x7x8x8x8x8'
    export PAID_KEY = 'X6x5x7x8x6c6c7x8x76c6c8x8c7'
    export PAID_SECRET = 'Xxx78x7x7x8x78x7x7x8x8x8x8'
    
    Name the hidden file as '.env'.
    Do not include paid key and secret if you do not have them.
    Put the '.env' file in the same folder as your program.
    
    Return
    ------
    tuple: (FREE_KEY,FREE_SECRET,PAID_KEY,PAID_SECRET)
    '''
    config = dotenv_values(location)
    FREE_KEY = config['FREE_KEY']
    PAID_KEY = config['PAID_KEY']
    
    try:
        FREE_SECRET = config['FREE_SECRET']
        PAID_SECRET = config['PAID_SECRET']
    except:
        FREE_SECRET = None
        PAID_SECRET = None
        
    return (FREE_KEY,FREE_SECRET,PAID_KEY,PAID_SECRET)

def Get_FacePlusPlus_Outputs(img,KEY,SECRET,landmark_106=2,compare_face=1):
    '''
    Params
    ------
    img          : numpy array of shape (n,n,3)
    landmark_106 : 2 if 106-D landmarks
    compare_face : 1 or any number
                   if face_num>compare_face:
                       return 0,0,0,0
                   if there are more than one face,
                   then it is not what we want. However, if 
                   you know that the face you want is bigger
                   than all other faces (i.e. in the foreground),
                   you can set compare face to be more than 5
                   so that the program can compare all faces and 
                   return only the biggest face.
    Return
    ------
    Features of only one face:
    
    (score, landmarks, attributes, rectangle) if face detected
    (0, 0, 0, 0) if no face detected
    '''
    
    # Convert img from array to string then to base64
    img_str = cv2.imencode('.jpg',img)[-1]
    img_64 = base64.b64encode(img_str)
    
    # Prepare to drop the package to face++
    http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect';
    re_att = 'gender,age,smiling,headpose,eyestatus,emotion,ethnicity,eyegaze,beauty,mouthstatus,blur'
    payload = {'api_key': KEY,
               'api_secret': SECRET,
               'image_base64': img_64,
               'return_landmark': landmark_106,
               'return_attributes': re_att}
   
    try:
        # Request Face++, add timeout because it can get stuck
        response = requests.post(http_url, data=payload, timeout=5)
        face_num = response.json()['face_num']
        response = response.json()['faces']

        distances = {}
        responses = []
        face_count = 0

        # drop image where face_num larger than compare_face=1
        if face_num>compare_face:
            return 0,0,0,0
        else:
            # Face++ returns a list of dictionaries
            for face in response:

                # Since face++ only returns the largest 5 faces
                if face_count<5:

                    # Make response into a better format for landmarks and attributes
                    rectangle = face['face_rectangle']
                    landmarks = {i:(v['x'],v['y']) for i,v in face['landmark'].items()}
                    attributes = Get_Flattened_Dict(face['attributes'])
                    
                    # Check size of face by getting the length between eyes
                    distance = int(Get_Euclidean_Distance(
                        np.asarray(landmarks['left_eye_left_corner']),
                        np.asarray(landmarks['right_eye_right_corner'])))

                    score = (attributes['mouthstatus_other_occlusion'] <= 50) + \
                    ((attributes['eyestatus_left_eye_status_occlusion'] <= 50) or \
                     (attributes['eyestatus_right_eye_status_occlusion'] <= 50)) + \
                    (abs(attributes['headpose_pitch_angle']) <= 10) + \
                    (abs(attributes['headpose_yaw_angle']) <= 15) + \
                    (distance >= 40)

                    # Append distance into distances dict
                    distances[face_count]=distance

                    # Append all into a list 
                    responses.append((score,landmarks,attributes,rectangle))

                    # Face count adds one because 
                    # I don't want it to read the no.6 and more 
                    # Face because it returns nothing
                    face_count+=1

            # Finally, if no distance detected
            if len(distances)==0:
                return 0,0,0,0
            # If there is at least one distance
            else:
                # Check the maximum distance and return position of key
                selected_key = max(distances.items(), 
                                   key=operator.itemgetter(1))[0]

                # Unpack the best face using selected key position
                final_score = responses[selected_key][0]
                final_landmarks = responses[selected_key][1]
                final_attributes = responses[selected_key][2]
                final_rectangle = responses[selected_key][3]

                # Return score, landmarks, attributes,rectangle in this order
                return final_score, final_landmarks, final_attributes, final_rectangle

    # If for some reason unsuccessful
    # E.g. timeout after 2 seconds
    # Or no 'faces' key found in response
    except:
        # Return nothing
        return 0,0,0,0
    
def get_rectangle(rectangle):
    return [((rectangle['left'],
              rectangle['top']),
             (rectangle['left']+rectangle['width'],
              rectangle['top'])),
            ((rectangle['left'],
              rectangle['top']+rectangle['height']),
             (rectangle['left']+rectangle['width'],
              rectangle['top']+rectangle['height']))]
    
def get_faceplusplus_outputs(img,key,verify_score=5):
    '''
    Load the key and secrets using load_api_keys(location) first.
    This function calls the Get_FacePlusPlus_Outputs function. 
    This function allows you to request from Face++ using free key,
    before trying with the paid key to save $$$.
    
    Note that free key's priority is low and there is a limit. 
    
    Params
    ------
    img : numpy array of shape (n,n,3)
    key : API key in tuple (FREE_KEY,FREE_SECRET,PAID_KEY,PAID_SECRET)
    
    Return
    ------
    (score, landmarks, attributes, rectangle, key)
    (0, 0, 0, 0, key) if no face detected
    '''
    
    FREE_KEY,FREE_SECRET,PAID_KEY,PAID_SECRET = key
    
    score,landmarks,attributes,rectangle = \
    Get_FacePlusPlus_Outputs(img,FREE_KEY,FREE_SECRET,2,1)
    if landmarks != 0: 
        key = 'free | face detected'
        if score >= verify_score:
            return score,landmarks,attributes,get_rectangle(rectangle),key
        else: return score,0,0,0,'free | face not detected'
    else:
        if (PAID_KEY is not None) & (PAID_SECRET is not None):
            key = 'paid | face detected'
            score,landmarks,attributes,rectangle = \
            Get_FacePlusPlus_Outputs(img,PAID_KEY,PAID_SECRET,2,1)
            if score >= verify_score:
                return score,landmarks,attributes,get_rectangle(rectangle),key
            else: return score,0,0,0,'paid | face not detected'
        else: return score,0,0,0,'no paid key | face not detected'

def get_rotated_image(img,src_lt,src_rt,dst_lt=(33,33),
                      dst_rt=(191,33),imsize=224):
    '''
    The default destination points would put all the face
    in the same area given a image size of 224x224 pixels
    Note that the default points would put the face at the
    very center, covering exactly 50% of the pixels.
    The default point is calculated using the formula:
    length = int(np.sqrt((224*224)*0.5))
    pt1=(int(224/2-length/2),int(224/2-length/2))
    pt2=(int(224/2+length/2),int(224/2+length/2))
    '''
    inPts = [tuple(src_lt),tuple(src_rt)]
    outPts = [tuple(dst_lt),tuple(dst_rt)]
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180) 
    xin = c60*(
        inPts[0][0]-inPts[1][0])-s60*(
        inPts[0][1]-inPts[1][1])+inPts[1][0]
    yin = s60*(
        inPts[0][0]-inPts[1][0])+c60*(
        inPts[0][1]-inPts[1][1])+inPts[1][1]
    inPts.append((int(xin),int(yin)))
    xout = c60*(
        outPts[0][0]-outPts[1][0])-s60*(
        outPts[0][1]-outPts[1][1])+outPts[1][0]
    yout = s60*(
        outPts[0][0]-outPts[1][0])+c60*(
        outPts[0][1]-outPts[1][1])+outPts[1][1]
    outPts.append((int(xout),int(yout)))
    tform = cv2.estimateAffine2D(
        np.array([inPts]),np.array([outPts]))[0]
    tform = np.float32(tform.flatten()[:6].reshape(2,3))
    return cv2.warpAffine(img,tform,(imsize,imsize))

def to_grayscale(img,bgr=True,vgg=True):
    '''
    convert to grayscale
    default assumes rgb color image
    
    returns
    =======
    img in three dimensional array
    '''
    conversion = [0.1140,0.5870,0.2989] if \
    bgr==True else [0.2989,0.5870,0.1140]
    gray = np.dot(img[...,:3],conversion)
    if vgg == True:
        gray = np.clip(np.stack(
            [gray,gray,gray],
            axis=-1),0,1)
    return gray

    
    
    
    
from statsmodels.discrete.discrete_model import MNLogit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def Classify_LR(df,yvar='type',method='Logistic Regression'):
    CATS=dict(df[yvar].value_counts()).keys()
    CATS=['Grp_{}'.format(int(i)) for i in CATS];CATS=sorted(CATS)
    
    clf1_md=LogisticRegression(max_iter=1000,solver='newton-cg',multi_class='auto')
    clf2_md=RandomForestClassifier(n_estimators=100,criterion='entropy')
    clf3_md=GaussianNB();clf1_nm='Logistic Regression'; 
    clf2_nm='Random Forest Classifier';clf3_nm='Gaussian Naive Bayes'
    clfs_dt=dict(zip((clf1_nm,clf2_nm,clf3_nm),(clf1_md,clf2_md,clf3_md)))
    
    clf=clfs_dt[method];X=df.drop(yvar,axis=1).values;y=df[yvar].values
    clf.fit(X,y);y_scores=clf.predict_proba(X);y_h=clf.predict(X)
    y_t=y.reshape(len(y),-1);accuracy=clf.score(X,y)
    y_scores=pd.DataFrame(y_scores,columns=CATS,index=df.index)
    return y_scores,y_t,accuracy

def logistic_regression(df,printing=False):
    y=df['type'].values
    ohe=OneHotEncoder(categories='auto',sparse_output=False)
    y_bin=ohe.fit_transform(y.reshape(len(y),-1))
    X=df.drop(['type'],axis=1)
    mod=MNLogit(y_bin,X);res=mod.fit()
    if printing==True:
        print(res.summary())
    return res

def match_df(df,scores,method='Propensity',yvar='type',threshold=0.0001):
    if method=='Euclidean':
        Prop_Names=['Prop_Score_{}'.format(j) 
                    for j in list(set(df[yvar].values.astype(int)))]
        df_new=df.copy();
        group_all=dict(df[yvar].value_counts())
        group_min=min(group_all.items(), key=operator.itemgetter(1))[0]
        group_trt=list(set(group_all.keys()).difference({group_min}))
        final=[];drop=[];count=1;total=len(df_new[df_new[yvar]==group_min])
        control=df_new[df_new[yvar]==group_min]
        for i,v in control.iterrows():
            temp_sel={}
            for group in group_trt:
                test=np.square(df_new[df_new[yvar]==group].drop('type',axis=1)-v.drop('type'))
                test=pd.DataFrame(test.sum(axis=1),index=test.index,columns=['score'])
                test_sel=np.sqrt(test).sort_values('score')
                name_sel=test_sel.index.values[0]
                eucl_sel=test_sel.values[0]
                temp_sel[name_sel]=eucl_sel
                df_new=df_new.drop(name_sel)
            if max(list(temp_sel.values()))<threshold:
                final+=list(temp_sel.keys())+[i]
            else:
                drop.append(i)
            print(f'Matching attributes by image: ({count}/{total})... Pruned: {len(drop)}',
                  end='\r');count+=1                
    elif method=='Propensity':
        Prop_Names=['Prop_Score_{}'.format(j) 
                    for j in list(set(df[yvar].values.astype(int)))]
        scores.columns=Prop_Names; 
        df=df.join(scores);
        df_new=df.copy();
        group_all=dict(df[yvar].value_counts())
        group_min=min(group_all.items(),key=operator.itemgetter(1))[0]
        group_trt=list(set(group_all.keys()).difference({group_min}))
        final=[];drop=[];count=1;total=len(df_new[df_new[yvar]==group_min])
        control=df_new[df_new[yvar]==group_min]
        for i,v in control.iterrows():
            temp_sel={}
            for group in group_trt:
                test=df_new[df_new[yvar]==group][Prop_Names]-v[Prop_Names]
                test_sel=abs(test).mean(axis=1).sort_values()
                name_sel=test_sel.index.values[0]
                prop_sel=test_sel.values[0]
                temp_sel[name_sel]=prop_sel
                df_new=df_new.drop(name_sel)
            if max(list(temp_sel.values()))<threshold:
                final+=list(temp_sel.keys())+[i]
            else:
                drop.append(i)
            print(f'Matching attributes by image: ({count}/{total})... Pruned: {len(drop)}',
                  end='\r');count+=1

    else:
        print('Please specify a method...')
    print()
    return list(set(final))

# load the attributes
def load_data(sex,N=None):
    
    # load groundtruths
    def load_gts(sex):
        gts = pd.DataFrame([[i.split('_')[2],
                             i.split('_')[0],
                             i.split('_')[1]] \
               for i in os.listdir('./tinder-scraper/images/download/') \
               if len(i)>20])
        gts = gts.rename({0:'index',1:'age',2:'group'},axis=1)
        gts['age']       = gts[['age']].astype(int)
        gts['gender']    = gts.apply((lambda x: 1 if 'woman' in x['group'] else 0),axis=1)
        gts['sexuality'] = gts.apply((lambda x: 0 if 'straight' in x['group'] else 1),axis=1)
        gts = gts[(gts['age'] >= 18) & (gts['age'] <= 40)].set_index('index')

        return gts[gts['gender']==0] if sex =='man' else gts[gts['gender']==1]
    
    gts = load_gts(sex)
    attributes = {}
    facial_images = {}
    
    for cat in ['gayman','straightman'] if sex =='man' else ['lesbianwoman','straightwoman']:
        
        attributes[cat] = {} ; count = 1
        src_dir = f'./Data/cleaned/{cat}/'
    
    #######   #######   #######   #######   #######   #######   #######   #######           
    
        for idx in os.listdir(src_dir)[:N]:
    
    #######   #######   #######   #######   #######   #######   #######   #######               

            print(f'Loading {cat} data: ({count}/{len(os.listdir(src_dir))})',
                  end='\r') ; count+=1

            for img in os.listdir(src_dir+idx):

                with open(src_dir+idx+'/'+img,'rb') as f:
                    loaded = pickle.load(f)
                
                facial_images[f'{cat}/{idx}/{img}'] = loaded['img']
                attribute_data = loaded['attributes']

                if attribute_data != 0:

                    attribute_data.update({'Age':gts.loc[idx]['age']})
                    attribute_data.update({'Sexuality':gts.loc[idx]['sexuality']})
                    attribute_data.update({'Image':int(img.split('.pickle')[0])-1})
                    attributes[cat][f'{cat}/{idx}/{img}'] = attribute_data
            
    return attributes,facial_images

def clean_attributes(final_data):
    
    def clean(df):
        df=df.dropna(axis=0,how='any');
        df['Beauty']=(df['beauty_female_score']+df['beauty_male_score'])//2
        df['Neutral']=df.apply((lambda x:0 if x['emotion_neutral']<50 else 1),axis=1)
        df['Anger']=df.apply((lambda x:0 if x['emotion_anger']<50 else 1),axis=1)
        df['Surprise']=df.apply((lambda x:0 if x['emotion_surprise']<50 else 1),axis=1)
        df['Disgust']=df.apply((lambda x:0 if x['emotion_disgust']<50 else 1),axis=1)
        df['Sadness']=df.apply((lambda x:0 if x['emotion_sadness']<50 else 1),axis=1)
        df['Happiness']=df.apply((lambda x:0 if x['emotion_happiness']<50 else 1),axis=1)
        df['glass_value']=df[['glass_value']].replace({'None':0,'Normal':1,'Dark':1})
        df['Glasses']=df['glass_value']
        df['Eyes']=df[['eyestatus_left_eye_status_no_glass_eye_close',
                       'eyestatus_left_eye_status_normal_glass_eye_close',
                       'eyestatus_right_eye_status_no_glass_eye_close',
                       'eyestatus_right_eye_status_normal_glass_eye_close']].sum(axis=1).astype(int)
        df['Eyes']=df.apply((lambda x:0 if x['Eyes']<1 else 1),axis=1)
        df['Smiling']=(df['smile_value'])
        df['Roll']=df['headpose_roll_angle']
        df['Yaw']=df['headpose_yaw_angle']
        df['Pitch']=df['headpose_pitch_angle']
        df=df[['Age',
               'Sexuality',
               'Image',
               'Beauty',
               'Neutral',
               'Happiness',
               'Anger',
               'Surprise',
               'Disgust',
               'Sadness',
               'Eyes',
               'Glasses',
               'Smiling',
               'Roll',
               'Yaw',
               'Pitch']]
        df.index=df.index.astype(str)

        return df.astype(int)

    dfs = []
    for i in final_data.keys():
        print(f'Cleaning {i}')
        convert_age = pd.DataFrame.from_dict(final_data[i],orient='index')        
        convert_age['Age'] = convert_age.apply((lambda x:x['Age'] if isinstance(x['Age'],np.int64) \
                                                else x['Age'][1]),axis=1)
        convert_age['Sexuality'] = convert_age.apply((lambda x:x['Sexuality'] if \
                                                      isinstance(x['Sexuality'],np.int64)
                                                      else x['Sexuality'][1]),axis=1)
        df_ = clean(convert_age)
        df_['type'] = 0 if 'straight' in i else 1
        dfs.append(df_)
        
    return pd.concat(dfs)

def split_attributes(sex,df,test_size=.15,final_dfs={},predictor=''):
    
    train_loc,test_loc = train_test_split(df[df['Image']==0].index.tolist(),
                                          test_size=test_size)
    
    for split,locs in {'train':train_loc,'test':test_loc}.items():
        idx_u = []
        for i in locs:
            for j in range(1,11):
                idx_u.append(i.split('1.pickle')[0]+f'{j}.pickle')
                
        final_dfs[split] = df.loc[set(df.index).intersection(set(idx_u))]
    
    print('Split into train-test sets')
    
    with open(f'./Data/test_set_{sex}_{predictor}.pickle','wb') as f:
        pickle.dump(final_dfs['test'].index.tolist(),f)
    print('Dumped test set, returning train set')
    return final_dfs['train']

def match_age(attributes):
    
    def match(df,yvar='type'):
        group_all = dict(df[yvar].value_counts())
        min_group = min(group_all.items(),key=operator.itemgetter(1))[0]
        max_group = max(group_all.items(),key=operator.itemgetter(1))[0]
        if min_group == max_group:
            min_group = 1; max_group = 0
        group_ctl = df[df[yvar]==min_group].sample(frac=1,random_state=1)
        group_trt = df[df[yvar]==max_group].sample(frac=1,random_state=1)

        final = []
        drop  = []
        count = 1
        total = len(group_ctl)

        for i,v in group_ctl.iterrows():

    #######   #######   #######   #######   #######   #######   #######   #######   

            selected = abs(group_trt[['Age']]-v['Age'])<=1

    #######   #######   #######   #######   #######   #######   #######   #######   

            selected = selected[selected['Age']==True]
            if selected.shape[0]>0:
                selected_index = selected.index[0]
                group_trt.drop(selected_index,axis=0,inplace=True)
                final+=[selected_index]+[i]
            else: 
                drop.append(i)

            print('Matching age by individual: ({}/{})... Pruned: {}'.format(count,total,len(drop)),
                          end='\r');count+=1
        print()
        return list(set(final))
    
    locs = match(attributes[attributes['Image']==0])
    idx_u = []
    for i in locs:
        for j in range(1,11):
            idx_u.append(i.split('1.pickle')[0]+f'{j}.pickle')

    return attributes.loc[set(attributes.index).intersection(set(idx_u))]

def match_attributes(df,threshold=0.0001):
    
    y_scores,_,_ = Classify_LR(df)
    df_b = df.loc[match_df(df,y_scores,threshold=threshold)]
    df_a = pd.concat([df[df['type']==0].sample(df_b.type.value_counts()[0]),
                      df[df['type']==1].sample(df_b.type.value_counts()[1])])
    
    df_a = df_a.sample(frac=1)

    _,_,scores_a = Classify_LR(df_a)
    _,_,scores_b = Classify_LR(df_b)
    
    print(f'Acc of Train Set A: {round(scores_a,3)} | N: {len(df_a)}')
    print(f'Acc of Train Set B: {round(scores_b,3)} | N: {len(df_b)}')

    return df_a,df_b





def get_dist_angle(X):
    
    distance = []
    angle = []
    for i in itertools.combinations(X,2):
        distance.append(Get_Euclidean_Distance(i[0],i[1])/224)
        angle.append(np.rad2deg(np.arctan((abs(i[0][1]-i[1][1]+2e-52))/\
                                          (abs(i[0][0]-i[1][0]+2e-52))))/180)
        
    return distance + angle





def augment_img(temp,cat,
                augment_type='baseline',
                augment_value=0,bgr=True):
    '''
    Params
    ------
    img          : numpy array of shape (224,224,3),
                   either normalized (0 to 1) or not (0 to 255).
    cat          : REQUIRED for blur & augment_value==2
                   because it is used to load the mean rgb value
                   of the dataset from pickle.
    augment_type : ['baseline','mask','border','blur','pixel']
    augment_value: 0 to 1 continuous variable
    bgr          : VGG-Face is trained using BGR images.
                   Thus, it is important to return the right
                   dimension but for plotting graphs, return
                   the RGB image by setting bgr = False.
    Return
    ------
    img in the shape and format: 
        image shape = [height,width,color channels]
        color channels = [blue,green,red]
    '''
    img = temp.copy()
    if (np.max(img)<=1):
        img = img.astype(float)
    elif (np.max(img)>1):
        img = (img/255).astype(float)
    
    # initiate the mask
    length = int(np.sqrt((224*224)*augment_value))
    pt1 = (int(img.shape[0]/2-length/2),int(img.shape[0]/2-length/2))
    pt2 = (int(img.shape[0]/2+length/2),int(img.shape[0]/2+length/2))
    mask = cv2.rectangle(np.zeros(img.shape),pt1,pt2,(255,255,255),-1)/255
    
    # initiate the noise
    noise = np.clip(np.random.normal(0.5,0.1,(224,224,3)),0,1)
    
    if augment_type=='mask':
        if augment_value==0:
            pass
        else:
            img = np.clip(img*(1-mask),0,1)
            
    elif (augment_type=='border'):
        if augment_value==0:
            pass
        else:
            img = np.clip(img*(mask),0,1)
        
    elif (augment_type=='pixel'):
        if augment_value==0:
            pass
        else:
            img = img*(1-augment_value)+noise*(augment_value)
            
    elif (augment_type=='blur'):
        k = int(augment_value*30)
        b_deg = [224,112,74,56,44,37,32,28,24,22,20,19,18,17,
                 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,1]
        if k==0:
            pass
        elif 1<=k<30:
            img = cv2.resize(img,(b_deg[k],b_deg[k]),interpolation=cv2.INTER_AREA)
            img = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
        elif k==30:
            img = np.ones((224,224,3))*(np.array([dataset_rgb[cat]['mu']]))  
    else:
        pass
    
    if bgr==True:
        # convert rgb to bgr because vggface model was trained using bgr
        img = cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
        img = img/255

    return np.clip(img,0,1)





def load_custom_vgg(model_fp='./models/vggface.h5'):
    """Build the VGGFace feature extractor. Requires TensorFlow/Keras
    (install with: pip install face_utils[deep])."""
    try:
        import tensorflow.keras.backend as K
        from tensorflow.keras import layers
        from tensorflow.keras.models import Model
    except ImportError as e:
        raise ImportError(
            "load_custom_vgg requires TensorFlow. Install it with "
            "`pip install face_utils[deep]` or `pip install tensorflow`."
        ) from e
    K.clear_session()

    # Input tensor
    I = layers.Input(shape=(224,224,3))

    # Block 1
    x = layers.Convolution2D(64,(3,3),activation='relu',padding='same',name='conv1_1')(I)
    x = layers.Convolution2D(64,(3,3),activation='relu',padding='same',name='conv1_2')(x)
    x = layers.MaxPooling2D((2,2),strides=(2,2),name='pool1')(x)

    # Block 2
    x = layers.Convolution2D(128,(3,3),activation='relu',padding='same',name='conv2_1')(x)
    x = layers.Convolution2D(128,(3,3),activation='relu',padding='same',name='conv2_2')(x)
    x = layers.MaxPooling2D((2,2),strides=(2,2),name='pool2')(x)

    # Block 3
    x = layers.Convolution2D(256,(3,3),activation='relu',padding='same',name='conv3_1')(x)
    x = layers.Convolution2D(256,(3,3),activation='relu',padding='same',name='conv3_2')(x)
    x = layers.Convolution2D(256,(3,3),activation='relu',padding='same',name='conv3_3')(x)
    x = layers.MaxPooling2D((2,2),strides=(2,2),name='pool3')(x)

    # Block 4
    x = layers.Convolution2D(512,(3,3),activation='relu',padding='same',name='conv4_1')(x)
    x = layers.Convolution2D(512,(3,3),activation='relu',padding='same',name='conv4_2')(x)
    x = layers.Convolution2D(512,(3,3),activation='relu',padding='same',name='conv4_3')(x)
    x = layers.MaxPooling2D((2,2),strides=(2,2),name='pool4')(x)

    # Block 5
    x = layers.Convolution2D(512,(3,3),activation='relu',padding='same',name='conv5_1')(x)
    x = layers.Convolution2D(512,(3,3),activation='relu',padding='same',name='conv5_2')(x)
    x = layers.Convolution2D(512,(3,3),activation='relu',padding='same',name='conv5_3')(x)
    x = layers.MaxPooling2D((2,2),strides=(2,2),name='pool5')(x)

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, name='fc6')(x)
    x = layers.Activation('relu', name='fc6/relu')(x)
    x = layers.Dense(4096, name='fc7')(x)
    x = layers.Activation('relu', name='fc7/relu')(x)
    x = layers.Dense(2622, name='fc8')(x)
    O = layers.Activation('softmax', name='fc8/softmax')(x)

    # Create the original model
    vggface = Model(I,O)
    vggface.load_weights(model_fp)
    vggface.trainable=False

    # return the custom VGG model
    return Model(inputs=vggface.input, outputs=vggface.get_layer('fc7/relu').output)

def load_custom_lr():
    # return the custom SVD and LR model
    return Pipeline(steps=[('svd', TruncatedSVD(n_components=500)),
                           ('lr', LogisticRegression(penalty='l1',
                                                     solver='liblinear'))],verbose=100)

