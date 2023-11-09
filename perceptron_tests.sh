#nohup python chapter_04_gw_perceptron.py > perceptron_0.txt &
#sleep 20s
nohup python chapter_04_gw_perceptron.py --layers 64 > perceptron_64.txt &
sleep 20s
nohup python chapter_04_gw_perceptron.py --layers 128 > perceptron_128.txt &
sleep 20s
nohup python chapter_04_gw_perceptron.py --layers 256 > perceptron_256.txt &
sleep 20s
nohup python chapter_04_gw_perceptron.py --layers 64 64 > perceptron_64_64.txt &
sleep 20s
nohup python chapter_04_gw_perceptron.py --layers 128 64 > perceptron_128_64.txt &
sleep 20s
nohup python chapter_04_gw_perceptron.py --layers 128 128 > perceptron_128_128.txt &
sleep 20s
nohup python chapter_04_gw_perceptron.py --layers 64 64 64 > perceptron_64_64_64.txt &