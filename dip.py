import streamlit as st
import cv2
import numpy as np
from math import floor,ceil
from matplotlib import pyplot as plt
from matplotlib import patches
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display

st.title("Watermarker")

def search(pixel,f,w):
  HT,WT = f.shape
  ht,wt = w.shape
  f = np.array(f)
  K = np.array([])
  # create an image of just the pixel, having the same size of original pixel
  pixel_tile = np.tile(pixel,(HT,WT))
  # absolute difference of the two images
  diff = np.abs(f - pixel_tile)
  result = np.argwhere(diff == 0)
  while len(result) == 0:
    i=1
    result = np.argwhere((diff>=-i) | (diff<=i))
    i = i + 1
  K1 = np.append(K,result[0][0])
  K2 = np.append(K1,result[0][1])
  K = K2
  return K

def invisible_insertion(f,w):
  L = np.array([]) 
  ht,wt = w.shape
  for l in range(wt):
    for p in range(ht):
      p = search(w[p,l],f,w)
      L = np.append(L,p)
  Key = np.array([])
  Key = np.append(Key,ht)
  Key = np.append(Key,wt)
  Key = np.append(Key,L)
  Key = Key.astype(int)
  return Key

def invisible_extract(image,key):
  m = int(key[0])
  n = int(key[1])
  v = 0
  C = np.array([0])
  R = np.array([0])
  watermark = np.zeros((m,n))
  for i in range(2,len(key)-1,2):
    if i == 2:
      C[0] = int(key[i])
      R[0] = int(key[i+1])
    else :
      C = np.append(C,int(key[i]))
      R = np.append(R,int(key[i+1]))
  for i in range(m-1):
    for j in range(n-1):
      watermark[i,j] = image[C[v],R[v]]
      v = v+1
  watermark = cv2.flip(watermark,0)
  M = cv2.getRotationMatrix2D((m/2, n/2), -90, 1.0)
  watermark = cv2.warpAffine(watermark, M, (m, n))
  return watermark

def visible_insertion(f,w,alpha,position = 'center'):
  f_row,f_column = f.shape
  w_row,w_column = w.shape
  g = f
  if position == 'center':
    row_start = f_row//2 - w_row//2
    row_end = f_row//2 + w_row//2
    col_start = f_column//2 - w_column//2
    col_end = f_column//2 + w_column//2
  elif position == 'top left':
    row_start = 0
    row_end =  w_row
    col_start = 0
    col_end =  w_column
  elif position == 'bottom left':
    row_start = f_row - w_row
    row_end = f_row 
    col_start = 0
    col_end = w_column
  elif position == 'top right':
    row_start = 0
    row_end = w_row
    col_start = f_column - w_column
    col_end = f_column 
  elif position == 'bottom right':
    row_start = f_row - w_row
    row_end = f_row  
    col_start = f_column - w_column
    col_end = f_column 

  g[row_start:row_end,col_start:col_end] = (1-alpha) * f[row_start:row_end,col_start:col_end] + alpha * w
  return g

def read_images(value):
  img = Image.open(value)
  opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
  return opencvImage

image = st.file_uploader("Upload Image",accept_multiple_files=False)
if image:
    image = read_images(image)
#st.write(image)
#st.write(type(image))
watermark = st.file_uploader("Upload Watermark",accept_multiple_files=False)
if watermark:
    watermark = read_images(watermark)
method = st.radio('Choose a method',['Visible','Invisible'])
alpha = st.slider("Select Alpha", min_value=0.0,max_value=1.0,step=0.01)
position = st.selectbox('Position of watermark',
            ['center','top right','top left', 'bottom left','bottom right'])
if st.button("Insert"):
    if np.any(image) and np.any(watermark):
        if method == 'Visible':
            g = visible_insertion(image, watermark,alpha,position)
            img_g = Image.fromarray(g)
            st.image(img_g)
        elif method == "Invisible":
            st.write('The Key is:')
            st.write(str(invisible_insertion(image,watermark)))
    else:
        st.write("Insert both Image and Watermark")