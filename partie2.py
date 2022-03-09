import cv2
import numpy as np
from tkinter import filedialog

print(' ** Projet vision M2 SII Partie 2 : Reconstruction 3D photométrique .............')
print(' .............')
direct = filedialog.askopenfilename(initialdir="/",title="selectionner le fichier light_intensities") #pour selectionner le chemin de fichier light_intensities 

def load_lightSources(lien):
    fichier=open(filedialog.askopenfilename(initialdir="/",title="selectionner le fichier light_directions"),'r',encoding='UTF-8')
    
    lines=fichier.readlines()    #lire le fichier ligne par ligne
    i=0
    h=len(lines)
    result= np.empty((h,3))
    while(i<len(lines)):
        result[i]=lines[i].split()   #sauvgarder le ficher lu dans une matrice de 3 dimenssions
        i=i+1
    print("Le fichier lightSources a été chargé avec succès")
    return result


def load_intensSources(lien):
    fichier=open(direct,'r',encoding='UTF-8') #le chemin de fichier light_intensities
    
    lines=fichier.readlines()   #lire le fichier ligne par ligne
    i=0
    h=len(lines)
    result= np.empty((h,3))
    while(i<len(lines)):
        result[i]=lines[i].split()   #sauvgarder le ficher lu dans une matrice de 3 dimenssions
        i=i+1
    print("Le fichier intensSources a été chargé avec succès")
    return result

def load_objMask(lien):
   
    image=filedialog.askopenfilename(initialdir="/",title="selectionner l'image de mask")
    img = cv2.imread(str(image),0)     #lire l'image
    if img is None : 
        print('image vide 1')
        exit(0)
    else :
        h,w = img.shape
        imgRes = np.zeros((h,w),np.uint8)     
        for y in range(h):
            for x in range(w):
                if(img[y,x]!=0):
                    imgRes[y,x]=1   #rendre l'image en binaire si le bit=0 le laisser a 0 sinon le rendre a 1
    print("L'image objMask a été chargé avec succès")
    return imgRes
    
def load_images(lien):
    fichier=open(filedialog.askopenfilename(initialdir="/",title="selectionner le fichier filnames"),'r',encoding='UTF-8')
    
    lines=fichier.readlines()
    result = np.empty((len(lines),313344))  #créer une matrice a h*w*c colonnes et 96 lignes
    i=0
    mat=load_intensSources('') 
    directory=filedialog.askdirectory()
    print("Le fichier filenames a été chargé avec succès")
    while(i<96):
        line=lines[i].split() 
        
        image=directory+'/'+line[0]
         
        flo= np.zeros((512,612,3),np.float32)
        img = cv2.imread(str(image),cv2.IMREAD_COLOR)
        if img is None : 
         print('image vide')
         exit(0)
        else :
            c=0
            h,w,c= img.shape
            
            nvg= np.empty((h,w),np.float32)
            temp=0
            for y in range(h):
             x=0
             for x in range(w):
                 z=0
                 for z in range(c):
                     flo[y,x,z]=img[y,x,z]/((65536-1)*mat[i,2-z])  #charger chaque bit le diviser par (2^16-1) pour le rendre dans [0,1] puis diviser chaque couleur par la colonne adequate dans la matrice source
                     
                     
                 nvg[y,x]=flo[y,x,0]*0.11+flo[y,x,1]*0.59+flo[y,x,2]*0.3  #convertir l'image vers le niveau de gris
                

            imageLigne=np.empty(h*w)
            cpt=0
            for y in range(h):
             for x in range(w):
                 imageLigne[cpt]=nvg[y,x]   #convertir la matrice nvg en vecteur 
                 ##print( imageLigne[cpt])
                 cpt=cpt+1

             result[i]=imageLigne   #stockage des vecteurs dans la matrice result
        print("l'image ",i,"a été traitée")    
        i=i+1
    return result

def calcul_needle_map():
    light_sources=load_lightSources('')
    obj_masques=load_objMask('')
    obj_images =load_images('')
    

    result=np.zeros((512,612,3),np.uint8)
    temp1=np.zeros(3,np.uint8) 
    
    i=0
    x=-1
    y=0
 
    pinv=np.linalg.pinv(light_sources)  #calculer le pseudo inverse de la matrice lightsource et le stoquer dans pinv
    while(i<512*612):
        if(i%612==0):  #si on a parcouru w colonne on incremente le i (ligne) et on rend ley (colonne) a 0 pour sauvegarder directement dans la matrice 3D
            x=x+1  
            y=0
        if(obj_masques[x,y]==1):
            
            temp=np.dot(pinv,obj_images [:,i])  #calculer le produit matriciel entre pinv et une colonne de la matrice des images
            
            N=np.sqrt(temp[0]**2+temp[1]**2+temp[2]**2)  #calculer le N pour la normalisation 
        
            temp[0]=temp[0]/N  
            temp1[2]=((temp[0]+1)/2)*255
            temp[1]=temp[1]/N
            temp1[1]=((temp[1]+1)/2)*255
            temp[2]=temp[2]/N
            temp1[0]=((temp[2]+1)/2)*255
            
            result[x,y]=temp1  #sauvgarder le calcule après normalisation dans la matrice result a 3 dimenssions
        
        y=y+1
        i=i+1
    
    return result








r=calcul_needle_map()
cv2.imshow("source", r )
cv2.imwrite('resultat',r)
cv2.waitKey(0)
cv2.destroyAllWindows()




