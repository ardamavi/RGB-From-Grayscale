# Arda Mavi
import sys
from PIL import Image
from classifier_procedure import *
from copy import deepcopy

def listToImg(list, size):
    # Saving RGB Photo
    # Create new photo:
    rgb_photo = Image.new('RGB', size, "white")
    # Put data to new_photo:
    rgb_photo.putdata(list)
    return rgb_photo

def getImgList(img):
    # Return image matrix list:
    img_list = []
    rgb_list = [[], [], []]
    # 0 -> grayscale & 1 -> RGB
    imgMode = True if img.mode == 'L' else False
    for h in range(img.size[1]):
        for w in range(img.size[0]):
            if imgMode:
                # Photo is grayscale
                img_list.append([img.getpixel((w,h))])
            else:
                # Photo is RGB (for training:):
                pixels = img.getpixel((w,h))
                for layer in range(3):
                    rgb_list[layer].append(pixels[layer])
    return img_list if imgMode else rgb_list

def saveNewClassifiers(clfs):
    saveClassifier(clfs[0], 'Classifier/R_Classifier.pkl')
    saveClassifier(clfs[1], 'Classifier/G_Classifier.pkl')
    saveClassifier(clfs[2], 'Classifier/B_Classifier.pkl')

def prepareClassifiers(img, clf, grayPhotoList):
    imgs = getImgList(img)
    clfs = []
    rClf = trainClassifier(deepcopy(clf), grayPhotoList, imgs[0])
    clfs.append(rClf)
    gClf = trainClassifier(deepcopy(clf), grayPhotoList, imgs[1])
    clfs.append(gClf)
    bClf = trainClassifier(deepcopy(clf), grayPhotoList, imgs[2])
    clfs.append(bClf)
    return clfs

def createClassifier(img, grayPhotoList):
    # Creating classifier:
    # Train with same image:
    print('Classifier not found!\nCreating new classifier ...')
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    print('Classifier created!\nTraining classifier...')
    clfs = prepareClassifiers(img, clf, grayPhotoList)
    print('Classifier trained!\nTrained classifier saving...')
    saveNewClassifiers(clfs)
    print('Trained classifier saved in to "Classifier" folder!')
    return clfs

def rgbListsToList(lists):
    rgbList = []
    for i in range(len(lists[0])):
        rgbList.append((lists[0][i], lists[1][i], lists[2][i]))
    return rgbList


def grayscaleToRGB(img):
    print('Getting grayscale photo...')
    grayPhotoList = getImgList(img.convert('L'))
    # Getting classifier:
    clfs = getClassifiers()
    if clfs == None:
        clfs = createClassifier(img, grayPhotoList)
    print('Creating RGB photo...')
    rgbLists = getPredict(clfs, grayPhotoList)
    return rgbListsToList(rgbLists)

def main():
    # Getting starting argumans:
    start_arg = sys.argv
    # If starting argumans is empty, return error:
    if len(start_arg) < 2:
        print('Required argument:\n- (<PhotoFile>)')
        return

    # Getting photo name:
    photo_name = start_arg[1]
    # Getting photo:
    try:
        photo = Image.open(photo_name)
    except:
        # If photo not found:
        print('Photo not found!')
        return

    if photo.mode != 'L':
        # If image is not grayscale, create and save image with grayscale mode:
        gs_photo = photo.convert('L')
        gs_photo.save('GS_'+photo_name)
        print('Your RGB photo converted to grayscale and save as GS_'+photo_name)

    # Convert grayscale to RGB:
    rgb_photo_list = grayscaleToRGB(photo)

    # Get photo from list:
    rgb_photo = listToImg(rgb_photo_list, photo.size)
    print('Your RGB photo is ready!')
    # Save RGB Photo:
    rgb_photo.save('RGB_'+photo_name)
    print('Your RGB photo saved as RGB_'+photo_name)

    return


if __name__ == '__main__':
    main()
