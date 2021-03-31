import datetime

def convertTime(inputTime):

    # pass in time to this where: time = datetime.datetime(2017:04:01:0:0)
    if inputTime is " ":
        return "Invalid"
    elif inputTime.isalpha():
        return "Invalid format"
    else:
        date = []
        date = inputTime.split(":")
        y = int(date[0])
        m = int(date[1])
        d = int(date[2])
        h = int(date[3])
        mn = int(date[4])
        if y > 2018:
            return "Invalid"
        elif m > 12:
            return "Invalid"
        elif d > 31:
            return "Invalid"
        elif h > 60:
            return "Invalid"
        elif mn > 60:
            return "Invalid"
        time = datetime.datetime(y,m,d,h,mn).strftime('%s')
        return time


def main():
    time = str("2018:04:15:11:10")
    output = convertTime(time)
    print output
    time2 = str("2014:05:18:17:16")
    time3 = str("2019:13:33:25:76")
    output2 = convertTime(time2)
    output3 = convertTime(time3)
    print output2
    print output3
    blank = " "
    test = convertTime(blank)
    print test
    alpha = "abc"
    alpha_test = convertTime(alpha)
    print alpha_test


if __name__ == '__main__':
    main()
