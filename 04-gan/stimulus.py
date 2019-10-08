import numpy as np
import tensorflow as tf
from grid import bound

class Stimulus:
    def __init__(self, image, grid, xpos=0, ypos=0):
        self.shape = image.shape
        
        if len(self.shape) == 2:
            self.original = image.reshape(*self.shape, 1)
            self.shape = self.original.shape
        else:
            self.original = image
            
        # Normalise between -1 and 1 for an RGB255 image
        if np.max(self.original) > 1: 
            self.original = (self.original / 127.5) - 1
        
        self.padder = np.zeros((3 * self.shape[0], 3 * self.shape[1], self.shape[2])) - 1
        self.padder[self.shape[0]:2*self.shape[0], self.shape[1]:2*self.shape[1], :] = self.original
        
        self.xpos = xpos
        self.ypos = ypos
        
        self.image = self.getImage()
        
        self.grid = grid
        self.sampleWidth = 2
        
        self.vector = self.process()
            
    def get_params(self, x : float, y : float):
        
        ymin = bound(int(self.shape[0] * y - self.sampleWidth // 2), 0, self.shape[0] - 1)
        ymax = bound(int(self.shape[0] * y + self.sampleWidth // 2), 0, self.shape[0] - 1)
        xmin = bound(int(self.shape[1] * x - self.sampleWidth // 2), 0, self.shape[1] - 1)            
        xmax = bound(int(self.shape[1] * x + self.sampleWidth // 2), 0, self.shape[1] - 1)

        vals  = self.image[ymin:ymax, xmin:xmax, :]
        return np.mean(vals)
    
    def getImage(self):
        """ Based on xpos and ypos, get the image view from the padder.
        """
        
        xstart = self.shape[0] - int(self.xpos * self.shape[0])
        ystart = self.shape[1] - int(self.ypos * self.shape[1])
        
        return self.padder[ystart:ystart+self.shape[1], xstart:xstart+self.shape[0], :]

    def process(self):
        """ Converts the stimulus into a brightness vector for the
        """

        params = np.array([self.get_params(e.x, e.y) for e in self.grid.grid])
        # Normalise to between 0 and 1
        params = params - (np.min(params))
        if np.max(params) > 0:
            params /= np.max(params)
        return params
    
    def setPos(self, xpos: float, ypos: float):
        """Translate the image. xpos and ypos lie in the range (-1, 1)
        """
        self.xpos = xpos
        self.ypos = ypos
        self.image = self.getImage()
        self.vector = self.process()

class StimulusNet(Stimulus):

    def __init__(self, image, grid, encoder_path):
        self.encoder = tf.keras.models.load_model(encoder_path)      
        Stimulus.__init__(self, image, grid)
    
    def process(self):
        image_tensor = tf.convert_to_tensor(np.array([self.image]), dtype=tf.float32)
        return self.encoder(image_tensor).numpy()[0]
