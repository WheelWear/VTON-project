import csv

man_people = ['Jonghyeon','Seunghyuk','Mingun']
woman_people = ['Taeyoung','Seohyun']
man_top = ['manA','manB','manC','manD','manE','manF','manG','manH','manI']
man_bottom = ['mana','manb','manc','mand','mane','manf','mang','manh','mani']
woman_top = ['womanA','womanB','womanC','womanD','womanE','womanF']
woman_bottom = ['womana','womanb','womanc','womand','womane','womanf']
angle = ['30', '60', '90', '120', '150' ]

man_count = 0
with open('dataset_to_take.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename','조합명' ,'조합의 촬영 순서'])
    for man in man_people:
        for top in man_top:
            for bottom in man_bottom:
                for ang in angle:
                    filename = f'{man}_{top}_{bottom}_{ang}.jpg'
                    combination_name = f'{top}_{bottom}'
                    writer.writerow([filename, combination_name, None])
                    man_count += 1
    print("man_count :", man_count, end='\n')

    woman_count = 0
    for man in woman_people:
        for top in woman_top:
            for bottom in woman_bottom:
                for ang in angle:
                    filename = f'{man}_{top}_{bottom}_{ang}.jpg'
                    combination_name = f'{top}_{bottom}'
                    writer.writerow([filename, combination_name, None])
                    woman_count += 1
    print("woman_count :", woman_count)

