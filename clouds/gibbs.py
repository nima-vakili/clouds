import numpy as np

def get_state(params, attrs=('mu','Z','tau','R')):
#def get_state(params, attrs=('mu','tau','R')):

    state = []
    for attr in attrs:
        value = getattr(params, attr)
        if type(value) == np.ndarray:
            state.append(value.copy())
        else:
            state.append(value)
            
    return state

def set_state(params, state, attrs=('mu','Z','tau','R')):
#def set_state(params, state, attrs=('mu','tau','R')):
    
    for attr, value in zip(attrs, state):
        param = getattr(params, attr)
        if type(param) == np.ndarray:
            param[...] = value
        else:
            setattr(params, attr, value)
             
class GibbsSampler(object):

    def __init__(self, posteriors):
        self.pdfs = posteriors

    def run(self, x=None, n_iter=1):
        
        self.samples = samples = []
        params = self.pdfs[-1].params        	       

        if x is not None:
            set_state(params, x)

        for i in range(n_iter):  
            
            for pdf in self.pdfs:
                pdf.sample()
            samples.append(get_state(params))   
    
        return samples


        
        
        
        
