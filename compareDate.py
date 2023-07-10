from dateutil.parser import parse

def get_max_date(date1, date2):
    date1_obj = parse(date1)
    date2_obj = parse(date2)

    if date1_obj > date2_obj:
        return date1
    else:
        return date2
