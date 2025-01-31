file = open('compute_han_stats.sh', 'w')


for i in range(0, 23225, 25):
    file.write("python " + "compute_han_statistics.py " + str(i) + " " + str(i+25) + "\n")

file.close()

