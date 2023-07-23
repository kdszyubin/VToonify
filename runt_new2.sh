#!/bin/bash
cd /data/workspace/williamyang1991/VToonify

# 使用find命令来查找所有输入文件
content_array=($(find /data/dataset/input/vtoonify -type f \( -name "*.png" -o -name "*.jpeg" -o -name "*.jpg" \)))
ckpt_array=("./checkpoint/vtoonify_t_arcane/vtoonify.pt" "./checkpoint/vtoonify_t_caricature/vtoonify.pt" "./checkpoint/vtoonify_t_cartoon/vtoonify.pt" "./checkpoint/vtoonify_t_comic/vtoonify.pt" "./checkpoint/vtoonify_t_pixar/vtoonify.pt" "./checkpoint/vtoonify_t_illustration/vtoonify.pt")

for content in ${content_array[@]}
do
    for (( i=0; i<${#ckpt_array[@]}; i++ ))
    do
        ckpt=${ckpt_array[$i]}
        # 从style_id_arrays获取对应的style_id数组
	python style_transfer.py --content $content --scale_image --backbone toonify --ckpt $ckpt --padding 600 600 600 600 --cpu
        base=$(basename $content)
        base=${base%.jpeg}
        base=${base%.jpg}
        base=${base%.png}

        output_filename="${base}_vtoonify_t.jpg"
        # 从ckpt路径中提取模型名称
        model_name=$(basename $(dirname $ckpt))
        # 生成新的文件名
        output_filename_2="${base}_${model_name}.jpg"
        # 使用mv命令将输出文件移动到新的输出目录
        mv ./output/${output_filename} /data/dataset/output/vtoonify/${output_filename_2}
    done
done

