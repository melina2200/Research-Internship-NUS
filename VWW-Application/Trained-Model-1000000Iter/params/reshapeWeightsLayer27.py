#reshape params53
import numpy as np

with open('_param_53.txt') as f:
    lines = f.readlines()

params53 = []
for line in lines:
    numbers = line.split(',')
    for number in numbers:
        params53.append(int(number))
    print(len(params53))
    array = np.array(params53)
    array = np.reshape(array, (256,256))
    array1 = array[:,:32]
    array2 = array[:,32:64:]
    array3 = array[:,64:96]
    array4 = array[:,96:128]
    array5 = array[:,128:160]
    array6 = array[:,160:192]
    array7 = array[:,192:224]
    array8 = array[:,224:256]
    print(array1.shape)
    print(array2.shape)
    print(array3.shape)
    print(array4.shape)
    print(array5.shape)
    print(array6.shape)
    print(array7.shape)
    print(array8.shape)

    with open("param53_1.txt", "w") as f:
        f.write('weight layer 27, part 1  \n')
        np.savetxt(f, array1.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()

    with open("param53_2.txt", "w") as f:
        f.write('weight layer 27, part 2  \n')
        np.savetxt(f, array2.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()

    with open("param53_3.txt", "w") as f:
        f.write('weight layer 27, part 3  \n')
        np.savetxt(f, array3.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()

    with open("param53_4.txt", "w") as f:
        f.write('weight layer 27, part 4  \n')
        np.savetxt(f, array4.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()

    with open("param53_5.txt", "w") as f:
        f.write('weight layer 27, part 5  \n')
        np.savetxt(f, array5.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()

    with open("param53_6.txt", "w") as f:
        f.write('weight layer 27, part 6  \n')
        np.savetxt(f, array6.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()

    with open("param53_7.txt", "w") as f:
        f.write('weight layer 27, part 7  \n')
        np.savetxt(f, array7.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()

    with open("param53_8.txt", "w") as f:
        f.write('weight layer 27, part 8  \n')
        np.savetxt(f, array8.astype("int8"), newline=",",delimiter = ",", fmt='%d')
    f.close()