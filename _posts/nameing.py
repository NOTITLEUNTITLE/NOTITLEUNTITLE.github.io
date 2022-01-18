import os
from pathlib import Path
import cv2


path_dir = "/mnt/d/notitle-github-blog/NOTITLEUNTITLE.github.io/_posts/bandit/"
# path_dir = "/mnt/d/notitle-github-blog/NOTITLEUNTITLE.github.io/images/2022-01-18/"
# path_dir = r"D:\notitle-github-blog\NOTITLEUNTITLE.github.io\_posts\bandit"
# path_dir = r"D:\notitle-github-blog\NOTITLEUNTITLE.github.io\images\2022-01-18\\"

file_list = os.listdir(path_dir)
# print(Path.cwd())
# print(path_dir)
# print(file_list)
print(len(file_list))

basename1 = "2022-01-18-"
basename2 = "-bandit-"


i = 1
for file in file_list:
  src = os.path.join(path_dir, file)
  dst = basename1 + str(i) + basename2 + str(i-1) + '.md'
  dst = os.path.join(path_dir, dst)
  os.rename(src, dst)
  i += 1
  

# for name in file_names:
#     src = os.path.join(file_path, name)
#     dst = basename + str(i) + basename2 + str(i) + '.md'
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)
#     i += 1


# img_name = path_dir + file_list[0]
# img = cv2.imread(img_name, cv2.IMREAD_COLOR)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in range()
# os.path.join()


