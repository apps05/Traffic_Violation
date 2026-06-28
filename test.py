from solution import TrafficViolationDetector

model = TrafficViolationDetector('./models')

output = model.predict('test1.jpeg')

print(output)