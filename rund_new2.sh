#!/bin/bash
cd /data/workspace/williamyang1991/VToonify

# 使用find命令来查找所有输入文件
content_array=($(find /data/dataset/input/vtoonify -type f \( -name "*.png" -o -name "*.jpeg" -o -name "*.jpg" \)))
ckpt_array=("./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt" "./checkpoint/vtoonify_d_caricature/vtoonify_s_d.pt" "./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt" "./checkpoint/vtoonify_d_comic/vtoonify_s_d.pt" "./checkpoint/vtoonify_d_pixar/vtoonify_s_d.pt" "./checkpoint/vtoonify_d_illustration/vtoonify_s_d.pt")

# 定义多个一维数组，每个元素是一个style_id
style_id_arcane=(1 5 9)  # 第一个模型的style_id
style_id_caricature=(2 6 10) # 第二个模型的style_id
style_id_cartoon=(104 229 204 263 315) # 第三个模型的style_id
style_id_comic=(19 57 68 83) # 第四个模型的style_id
style_id_pixar=(16) # 第五个模型的style_id
style_id_illustration=(16 17 18) # 第六个模型的style_id

# 将style_id的数组放在另一个数组中
declare -A style_id_arrays=(
    ["0"]="${style_id_arcane[*]}"
    ["1"]="${style_id_caricature[*]}"
    ["2"]="${style_id_cartoon[*]}"
    ["3"]="${style_id_comic[*]}"
    ["4"]="${style_id_pixar[*]}"
    ["5"]="${style_id_illustration[*]}"
)

for content in ${content_array[@]}
do
    for (( i=0; i<${#ckpt_array[@]}; i++ ))
    do
        ckpt=${ckpt_array[$i]}
        # 从style_id_arrays获取对应的style_id数组
        style_ids=(${style_id_arrays[$i]})
        for style_id in ${style_ids[@]}
        do
            python style_transfer.py --content $content --scale_image --style_id $style_id --style_degree 0.5 --ckpt $ckpt --padding 600 600 600 600 --cpu
            base=$(basename $content)
            base=${base%.jpeg}
            base=${base%.jpg}
            base=${base%.png}

            output_filename="${base}_vtoonify_d.jpg"
            # 从ckpt路径中提取模型名称
            model_name=$(basename $(dirname $ckpt))
            # 生成新的文件名
            output_filename_2="${base}_${model_name}_$style_id.jpg"
            # 使用mv命令将输出文件移动到新的输出目录
            mv ./output/${output_filename} /data/dataset/output/vtoonify/${output_filename_2}
        done
    done
done

