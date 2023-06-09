import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

fig, ax = plt.subplots()
fig.set_tight_layout(False)
plt.subplots_adjust(left=0.25, bottom=0.25)
img_data = np.random.rand(50,50)
c_min = 0
c_max = 1

img = ax.imshow(img_data, interpolation='nearest')
cb = plt.colorbar(img)
axcolor = 'lightgoldenrodyellow'


ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_cmax  = plt.axes([0.25, 0.15, 0.65, 0.03])

s_cmin = Slider(ax_cmin, 'min', 0, 1, valinit=c_min)
s_cmax = Slider(ax_cmax, 'max', 0, 1, valinit=c_max)

def update(val, s=None):
    _cmin = s_cmin.val
    _cmax = s_cmax.val
    img.set_clim([_cmin, _cmax])
    plt.draw()

s_cmin.on_changed(update)
s_cmax.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    s_cmin.reset()
    s_cmax.reset()
button.on_clicked(reset)

plt.show()
