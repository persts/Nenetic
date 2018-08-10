import sys
import json

if len(sys.argv) < 3:
	print('Usage: merge .pnt_1 .pnt_2 ... .pnt_n')
	sys.exit()

merged = None
for f in range(1, len(sys.argv)):
	file = open(sys.argv[f], 'r')
	data = json.load(file)
	file.close()
	if merged is None:
		merged = data
	else:
		for image in data['points']:
			if image not in merged['points']:
				merged['points'][image] = {}
			for class_name in data['points'][image]:
				if class_name not in merged['classes']:
					merged['classes'].append(class_name)
					merged['classes'].sort()
					merged['colors'][class_name] = data['colors'][class_name]
				if class_name not in merged['points'][image]:
					merged['points'][image][class_name] = []
				merged['points'][image][class_name] = merged['points'][image][class_name] + data['points'][image][class_name]

file = open('merged.pnt', 'w')
json.dump(merged, file)
file.close()