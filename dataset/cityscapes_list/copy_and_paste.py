f = open("C:/Users/user/Desktop/personalstudy/auxiliary_classifier/dataset/cityscapes_list/train_foggy_0.005.txt", 'r')

g = open("C:/Users/user/Desktop/personalstudy/auxiliary_classifier/dataset/cityscapes_list/train_foggy_0.01.txt", 'w')

while True :
    line = f.readline()
    if not line :
        break
    first = line.split('/')[0]
    second = line.split('/')[1]
    third = second.split('_')[-1]
    new_second = second.split('_')[0]+'_'+second.split('_')[1]+'_'+second.split('_')[2]+'_'+second.split('_')[3]+'_'+second.split('_')[4]+'_'+second.split('_')[5]+'_'+'0.01.png\n'
    new_line = line.split('/')[0]+'/'+new_second
    g.write(new_line)

f.close()
g.close()
