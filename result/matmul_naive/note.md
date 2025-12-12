tinhanhnguyen@tinhanhnguyen-Z790-AORUS-ELITE-AX:~/Desktop/HK7/Parallel/Parallel_Computing/test$ g++ -O3 -march=native -fopenmp main.cpp -o main
tinhanhnguyen@tinhanhnguyen-Z790-AORUS-ELITE-AX:~/Desktop/HK7/Parallel/Parallel_Computing/test$ ./main 

===== Matrix Size: 100 x 100 =====
Testing naiveAddMultiply1 ...
Local time: trial 1 . Version: 1 Time Exe: 9.4542e-05s
Local time: trial 2 . Version: 1 Time Exe: 8.6734e-05s
Local time: trial 3 . Version: 1 Time Exe: 6.0759e-05s
Local time: trial 4 . Version: 1 Time Exe: 5.7088e-05s
Local time: trial 5 . Version: 1 Time Exe: 4.833e-05s
Local time: trial 6 . Version: 1 Time Exe: 4.8466e-05s
Local time: trial 7 . Version: 1 Time Exe: 4.8609e-05s
Local time: trial 8 . Version: 1 Time Exe: 4.8422e-05s
Local time: trial 9 . Version: 1 Time Exe: 4.8212e-05s
Local time: trial 10 . Version: 1 Time Exe: 4.8259e-05s
Avg time: 5.89421e-05s
Testing naiveAddMultiply2 ...
Local time: trial 1 . Version: 2 Time Exe: 4.6906e-05s
Local time: trial 2 . Version: 2 Time Exe: 4.6281e-05s
Local time: trial 3 . Version: 2 Time Exe: 4.6119e-05s
Local time: trial 4 . Version: 2 Time Exe: 4.9045e-05s
Local time: trial 5 . Version: 2 Time Exe: 4.6276e-05s
Local time: trial 6 . Version: 2 Time Exe: 4.629e-05s
Local time: trial 7 . Version: 2 Time Exe: 4.6407e-05s
Local time: trial 8 . Version: 2 Time Exe: 4.6422e-05s
Local time: trial 9 . Version: 2 Time Exe: 4.6228e-05s
Local time: trial 10 . Version: 2 Time Exe: 4.6338e-05s
Avg time: 4.66312e-05s
Testing naiveAddMultiply3 ...
Local time: trial 1 . Version: 3 Time Exe: 0.000277323s
Local time: trial 2 . Version: 3 Time Exe: 0.000178157s
Local time: trial 3 . Version: 3 Time Exe: 0.00018669s
Local time: trial 4 . Version: 3 Time Exe: 7.1224e-05s
Local time: trial 5 . Version: 3 Time Exe: 4.0475e-05s
Local time: trial 6 . Version: 3 Time Exe: 3.1192e-05s
Local time: trial 7 . Version: 3 Time Exe: 3.3433e-05s
Local time: trial 8 . Version: 3 Time Exe: 3.5104e-05s
Local time: trial 9 . Version: 3 Time Exe: 3.2506e-05s
Local time: trial 10 . Version: 3 Time Exe: 3.4256e-05s
Avg time: 9.2036e-05s

===== Matrix Size: 1000 x 1000 =====
Testing naiveAddMultiply1 ...
Local time: trial 1 . Version: 1 Time Exe: 0.0801412s
Local time: trial 2 . Version: 1 Time Exe: 0.0743222s
Local time: trial 3 . Version: 1 Time Exe: 0.0707995s
Local time: trial 4 . Version: 1 Time Exe: 0.0676592s
Local time: trial 5 . Version: 1 Time Exe: 0.0697588s
Local time: trial 6 . Version: 1 Time Exe: 0.0694863s
Local time: trial 7 . Version: 1 Time Exe: 0.0737148s
Local time: trial 8 . Version: 1 Time Exe: 0.0750547s
Local time: trial 9 . Version: 1 Time Exe: 0.0742621s
Local time: trial 10 . Version: 1 Time Exe: 0.069323s
Avg time: 0.0724522s
Testing naiveAddMultiply2 ...
Local time: trial 1 . Version: 2 Time Exe: 0.0693669s
Local time: trial 2 . Version: 2 Time Exe: 0.0690865s
Local time: trial 3 . Version: 2 Time Exe: 0.0723155s
Local time: trial 4 . Version: 2 Time Exe: 0.0746719s
Local time: trial 5 . Version: 2 Time Exe: 0.0745412s
Local time: trial 6 . Version: 2 Time Exe: 0.0749507s
Local time: trial 7 . Version: 2 Time Exe: 0.0687173s
Local time: trial 8 . Version: 2 Time Exe: 0.0677069s
Local time: trial 9 . Version: 2 Time Exe: 0.0695788s
Local time: trial 10 . Version: 2 Time Exe: 0.071064s
Avg time: 0.0712s
Testing naiveAddMultiply3 ...
Local time: trial 1 . Version: 3 Time Exe: 0.0737035s
Local time: trial 2 . Version: 3 Time Exe: 0.0721083s
Local time: trial 3 . Version: 3 Time Exe: 0.0650577s
Local time: trial 4 . Version: 3 Time Exe: 0.0618249s
Local time: trial 5 . Version: 3 Time Exe: 0.0673406s
Local time: trial 6 . Version: 3 Time Exe: 0.0595319s
Local time: trial 7 . Version: 3 Time Exe: 0.0707313s
Local time: trial 8 . Version: 3 Time Exe: 0.0715023s
Local time: trial 9 . Version: 3 Time Exe: 0.0746888s
Local time: trial 10 . Version: 3 Time Exe: 0.0642952s
Avg time: 0.0680785s

===== Matrix Size: 10000 x 10000 =====
Testing naiveAddMultiply1 ...
Local time: trial 1 . Version: 1 Time Exe: 171.409s
Local time: trial 2 . Version: 1 Time Exe: 169.617s
Local time: trial 3 . Version: 1 Time Exe: 171.435s
Local time: trial 4 . Version: 1 Time Exe: 171.307s
Local time: trial 5 . Version: 1 Time Exe: 171.101s
Local time: trial 6 . Version: 1 Time Exe: 171.238s
Local time: trial 7 . Version: 1 Time Exe: 171.617s
Local time: trial 8 . Version: 1 Time Exe: 172.025s
Local time: trial 9 . Version: 1 Time Exe: 172.752s
Local time: trial 10 . Version: 1 Time Exe: 170.868s
Avg time: 171.337s
Testing naiveAddMultiply2 ...
Local time: trial 1 . Version: 2 Time Exe: 172.006s
Local time: trial 2 . Version: 2 Time Exe: 170.876s
Local time: trial 3 . Version: 2 Time Exe: 172.487s
Local time: trial 4 . Version: 2 Time Exe: 171.578s
Local time: trial 5 . Version: 2 Time Exe: 169.324s
Local time: trial 6 . Version: 2 Time Exe: 169.018s
Local time: trial 7 . Version: 2 Time Exe: 168.768s
Local time: trial 8 . Version: 2 Time Exe: 170.221s
Local time: trial 9 . Version: 2 Time Exe: 169.086s
Local time: trial 10 . Version: 2 Time Exe: 168.742s
Avg time: 170.211s
Testing naiveAddMultiply3 ...
Local time: trial 1 . Version: 3 Time Exe: 460.262s
Local time: trial 2 . Version: 3 Time Exe: 472.968s
Local time: trial 3 . Version: 3 Time Exe: 448.324s
Local time: trial 4 . Version: 3 Time Exe: 480.573s
Local time: trial 5 . Version: 3 Time Exe: 516.885s
Local time: trial 6 . Version: 3 Time Exe: 478.552s
Local time: trial 7 . Version: 3 Time Exe: 477.842s
Local time: trial 8 . Version: 3 Time Exe: 511.965s
Local time: trial 9 . Version: 3 Time Exe: 479.865s
Local time: trial 10 . Version: 3 Time Exe: 488.65s
Avg time: 481.589s
tinhanhnguyen@tinhanhnguyen-Z790-AORUS-ELITE-AX:~/Desktop/HK7/Parallel/Parallel_Computing/test$ 