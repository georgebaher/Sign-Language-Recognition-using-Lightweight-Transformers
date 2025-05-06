connection_pairs=[(1,2),(3,4),(5,6)]
angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(connection_pairs)
                                     for j, b in enumerate(connection_pairs)
                                     if i < j]

angle_labels_2=[]
for i in range(len(connection_pairs)):
    for j in range(i+1, len(connection_pairs)):
        angle_labels_2.append(f"Angle{{{connection_pairs[i]}-{connection_pairs[j]}}}")

print(angle_labels)
print(angle_labels_2)