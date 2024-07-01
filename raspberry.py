from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import time

# Create a serial interface
serial = spi(port=0, device=0, gpio=noop())

# Create a MAX7219 device
device = max7219(serial, cascaded=4, block_orientation=-90)

# Display a scrolling message
msg = "65123465 65100372 65133654"
while True:
    for i in range(len(msg) * 8 + device.width):
        with canvas(device) as draw:
            draw.text((device.width - i, -2), msg, fill="white")
        time.sleep(0.1)

        # Clear the display
        device.clear()