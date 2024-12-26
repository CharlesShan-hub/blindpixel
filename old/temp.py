from PIL import Image

def invert_image(image_path, output_path):
    # 打开图片
    with Image.open(image_path) as img:
        # 将图片转换为灰度
        bw_img = img.convert('L')
        # 反转图片
        inverted_img = Image.eval(bw_img, lambda x: 255 - x)
        # 保存图片
        inverted_img.save(output_path)

# 使用脚本
input_image_path = '/Users/kimshan/Public/project/blindpixel/fastblindnet/assets/house_mask.png'  # 替换为你的图片路径
output_image_path = '/Users/kimshan/Public/project/blindpixel/fastblindnet/assets/house_mask0.png'  # 输出图片的路径
invert_image(input_image_path, output_image_path)
