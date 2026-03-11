import warp as wp
import newton
wp.init()
builder = newton.ModelBuilder()
body = builder.add_body("test")
print("Attributes:")
for k in dir(builder):
    if k.startswith('body'): print(k)
