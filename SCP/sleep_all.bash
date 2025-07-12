#!/bin/bash
pid=3746147 # 替换 <你的进程ID> 为实际的进程ID

# 循环直到进程不再运行
while ps -p $pid > /dev/null; do
    sleep 1
done

echo "上个任务结束"

for seed in 1 2 3
do 
  echo "开始 $seed 任务。"
  echo "开始第1个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh caltech101 $seed >./result/train/caltech101$seed.log &
  pid1=$!
  wait $pid1

  echo "开始 $seed 任务。"
  echo "开始第2个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh dtd $seed >./result/train/dtd$seed.log &
  pid2=$!
  wait $pid2


  echo "开始 $seed 任务。"
  echo "开始第3个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh eurosat $seed >./result/train/eurosat$seed.log &
  pid3=$!
  wait $pid3

  echo "开始 $seed 任务。"
  echo "开始第3+1个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh fgvc_aircraft $seed >./result/train/fgvc_aircraft$seed.log &
  pid5=$!
  wait $pid5



  echo "开始 $seed 任务。"
  echo "开始第5个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh food101 $seed >./result/train/food101$seed.log &
  pid6=$!
  wait $pid6

  echo "开始 $seed 任务。"
  echo "开始第6个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh oxford_flowers $seed >./result/train/oxford_flowers$seed.log &
  pid7=$!
  wait $pid7


  echo "开始 $seed 任务。"
  echo "开始第7个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh oxford_pets $seed >./result/train/oxford_pets$seed.log &
  pid8=$!
  wait $pid8

  echo "开始 $seed 任务。"
  echo "开始第8个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh stanford_cars $seed >./result/train/stanford_cars$seed.log &
  pid9=$!
  wait $pid9

  echo "开始 $seed 任务。"
  echo "开始第9个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh sun397 $seed >./result/train/sun397$seed.log &
  pid10=$!
  wait $pid10

  echo "开始 $seed 任务。"
  echo "开始第10个任务。"
  nohup ./scripts/maple/base2new_train_maple.sh ucf101 $seed >./result/train/ucf101$seed.log &
  pid11=$!
  wait $pid11


  echo "开始 $seed 任务。"
  echo "开始第11个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh caltech101 $seed >./result/test/caltech101__$seed.log &
  pid5666661=$!
  wait $pid5666661


  echo "开始 $seed 任务。"
  echo "开始第12个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh dtd $seed >./result/test/dtd_$seed.log &
  pid5666662=$!
  wait $pid5666662

  echo "开始 $seed 任务。"
  echo "开始第13个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh eurosat $seed >./result/test/eurosat__$seed.log &
  pid5666663=$!
  wait $pid5666663

  echo "开始 $seed 任务。"
  echo "开始第14个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh fgvc_aircraft $seed >./result/test/fgvc_aircraft__$seed.log &
  pid5666664=$!
  wait $pid5666664

  echo "开始 $seed 任务。"
  echo "开始第15个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh food101 $seed >./result/test/food101__$seed.log &
  pid5666665=$!
  wait $pid5666665

  echo "开始 $seed 任务。"
  echo "开始第16个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh oxford_flowers $seed >./result/test/oxford_flowers_$seed.log &
  pid5123=$!
  wait $pid5123

  echo "开始 $seed 任务。"
  echo "开始第17个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh oxford_pets $seed >./result/test/oxford_pets__$seed.log &
  pid5124=$!
  wait $pid5124

  echo "开始 $seed 任务。"
  echo "开始第18个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh stanford_cars $seed >./result/test/stanford_cars_$seed.log &
  pid5125=$!
  wait $pid5125

  echo "开始 $seed 任务。"
  echo "开始第19个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh sun397 $seed >./result/test/sun397__$seed.log &
  pid5126=$!
  wait $pid5126


  echo "开始 $seed 任务。"
  echo "开始第20个任务。"
  nohup ./scripts/maple/base2new_test_maple.sh ucf101 $seed >./result/test/ucf101_$seed.log &
  pid5127=$!
  wait $pid5127



  echo "所有 nohup 命令已依次执行完成。"


done

echo "普通imagenet 第一部分"


echo "普通imagenet 第一部分"
echo "开始第1个任务。"
nohup ./scripts/maple/base2new_train_maple.sh imagenet 1 >./result/imagenet1.log &
pid991=$!
wait $pid991



echo "开始第2个任务。"
nohup ./scripts/maple/base2new_test_maple.sh imagenet 1 >./result/imagenet__1.log &
pid994=$!
wait $pid994


echo "开始第3个任务。"
nohup ./scripts/maple/base2new_train_maple.sh imagenet 2 >./result/imagenet2.log &
pid992=$!
wait $pid992

echo "开始第4个任务。"
nohup ./scripts/maple/base2new_test_maple.sh imagenet 2 >./result/imagenet__2.log &
pid995=$!
wait $pid995





echo "普通imagenet 第二部分"
echo "开始第5个任务。"
nohup ./scripts/maple/base2new_train_maple.sh imagenet 3 >./result/imagenet3.log &
pid993=$!
wait $pid993


echo "开始第6个任务。"
nohup ./scripts/maple/base2new_test_maple.sh imagenet 3 >./result/imagenet__3.log &
pid996=$!
wait $pid996

# ============================================================================================
# ============================================================================================
# ============================================================================================
# ============================================================================================
# ============================================================================================


# for seed in 1 2 3 

# do
#   echo "XD开始seed $seed 任务。"


#   echo "开始第training个任务。"
#   nohup ./scripts/maple/xd_train_maple.sh imagenet $seed >./result/xd/aa_xd_imagenet_XX$seed.log &
#   pid990071=$!
#   wait $pid990071

#   echo "XD开始seed $seed 任务。"
#   echo "开始第1个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh caltech101 $seed >./result/xd/caltech101_XX$seed.log &
#   pid990072=$!
#   wait $pid990072

#   echo "XD开始seed $seed 任务。"
#   echo "开始第2个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh dtd $seed >./result/xd/dtd_XX$seed.log &
#   pid990073=$!
#   wait $pid990073


#   echo "XD开始seed $seed 任务。"
#   echo "开始第3个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh eurosat $seed >./result/xd/eurosat_XX$seed.log &
#   pid990074=$!
#   wait $pid990074

#   echo "XD开始seed $seed 任务。"
#   echo "开始第3+1个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh fgvc_aircraft $seed >./result/xd/fgvc_aircraft_XX$seed.log &
#   pid990075=$!
#   wait $pid990075



#   echo "XD开始seed $seed 任务。"
#   echo "开始第5个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh food101 $seed >./result/xd/food101_XX$seed.log &
#   pid990076=$!
#   wait $pid990076

#   echo "XD开始seed $seed 任务。"
#   echo "开始第6个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh oxford_flowers $seed >./result/xd/oxford_flowers_XX$seed.log &
#   pid990077=$!
#   wait $pid990077


#   echo "XD开始seed $seed 任务。"
#   echo "开始第7个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh oxford_pets $seed >./result/xd/oxford_pets_XX$seed.log &
#   pid990078=$!
#   wait $pid990078

#   echo "XD开始seed $seed 任务。"
#   echo "开始第8个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh stanford_cars $seed >./result/xd/stanford_cars_XX$seed.log &
#   pid990079=$!
#   wait $pid990079

#   echo "XD开始seed $seed 任务。"
#   echo "开始第9个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh sun397 $seed >./result/xd/sun397_XX$seed.log &
#   pid9900710=$!
#   wait $pid9900710

#   echo "XD开始seed $seed 任务。"
#   echo "开始第10个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh ucf101 $seed >./result/xd/ucf101_XX$seed.log &
#   pid9900711=$!
#   wait $pid9900711

#   echo "XD开始seed $seed 任务。"
#   echo "开始第11个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh imagenetv2 $seed >./result/xd/zz_imagenet_1v2_XX$seed.log &
#   pid9900712=$!
#   wait $pid9900712


#   echo "XD开始seed $seed 任务。"
#   echo "开始第12个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh imagenet_sketch $seed >./result/xd/zz_imagenet_2sketch_XX$seed.log &
#   pid9900713=$!
#   wait $pid9900713


#   echo "XD开始seed $seed 任务。"
#   echo "开始第13个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh imagenet_a $seed >./result/xd/zz_imagenet_3a_XX$seed.log &
#   pid9900714=$!
#   wait $pid9900714

#   echo "XD开始seed $seed 任务。"
#   echo "开始第14个任务。"
#   nohup ./scripts/maple/xd_test_maple.sh imagenet_r $seed >./result/xd/zz_imagenet_4r_XX$seed.log &
#   pid9900715=$!
#   wait $pid9900715

#   echo "所有 nohup 命令___$seed ___已依次执行完成。"

# done




echo "起床啦！！！！！！"
