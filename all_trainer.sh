python3 train.py -logfile 5test.txt -nnhs 1000 -bs 500 -eps 25
python3 test.py -logfile res_test5.txt -nnhs 1000 -bs 500 -eps 25
python3 train.py -logfile 51test.txt -nnhs 1000 -bs 500 -eps 25 -weisc 1
python3 test.py -logfile res_test51.txt -nnhs 1000 -bs 500 -eps 25 -weisc 1
python3 train.py -logfile 5test.txt -nnhs 1200 -bs 500 -eps 25 -weisc 1
python3 test.py -logfile res_test5.txt -nnhs 1200 -bs 500 -eps 25 -weisc 1
python3 train.py -logfile 51test.txt -nnhs 1200 -bs 500 -eps 25 -weisc 0.9
python3 test.py -logfile res_test51.txt -nnhs 1200 -bs 500 -eps 25 -weisc 0.9

