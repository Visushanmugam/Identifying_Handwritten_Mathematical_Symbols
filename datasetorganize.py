import os
import statistics

target_dir = 'dataset/valid'

sub = 0
to = []
for sub_dir in os.listdir(target_dir):
    i = 0
    for file in os.listdir(os.path.join(target_dir, sub_dir)):
        #if i > 550:
        #    #print(os.path.join(os.path.join(target_dir, sub_dir), file))
        #    os.remove(os.path.join(os.path.join(target_dir, sub_dir), file))
        i += 1
    
    sub += 1   
    to.append(i)    

    print(sub, i)    

val = statistics.median(to)

with open("model.txt", 'w') as modeljson:

    modeljson.write(str(os.listdir(target_dir)))



print(sub, val)