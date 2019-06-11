a = '/home/ubuntu/Desktop/text.txt'

f=open(a,'a')
for i in range (0 , 10):
	f.write('%d\n'%i)
