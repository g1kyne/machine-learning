from PIL import Image

def ImageToTxt(image_filename,txt_filename):
    '''将32*32像素的图片转化为只含01的txt文件'''
    image = Image.open(image_filename)
    # 显示图片
    #     im.show()
    width, height = image.size
    fh = open(txt_filename, 'w')
    for i in range(height):
        for j in range(width):
            # 获取像素点颜色
            color = image.getpixel((j, i))
            #RGB
            colorsum = color[0] + color[1] + color[2]
            #黑色RGB为0,0,0
            if colorsum == 0:
                fh.write('1')
            else:
                fh.write('0')
        fh.write('\n')
    fh.close()


#存储画图手写的数字，识别错误率较大
for i in range(10):
    ImageToTxt(str(i)+'_0.png', 'F:\\机器学习实战\\KNN\\testDigits'+str(i)+'_0.txt')
    print('存储了数字'+str(i)+'的txt文件')


