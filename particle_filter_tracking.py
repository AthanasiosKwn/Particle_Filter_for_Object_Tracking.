import cv2 as cv
import numpy as np



# Capture Video Object.
cap = cv.VideoCapture("o_sevenup.avi")

# Check to see if the video loaded or not.
if cap.isOpened():
    print("Video opened")
    
    # Video characteristics.
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object for output video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('try_output_video.avi', fourcc, fps, (width, height))

else:
    print("Video failed to load")
    exit()

# num of frames read
frames_read = 0

# Reading and processing frames
while True:

    ret, frame =cap.read()
    if not ret:
        break

    frames_read += 1

    if frames_read == 1:
    
        # The user manually defines the region of interest on the 1st frame (the position of the object of interest in frame 1).
        prior_roi = cv.selectROI('Image', frame, fromCenter=False, showCrosshair=True)
        x,y,w,h = prior_roi
        print("Selected ROI (x, y, w, h):", prior_roi)
        cv.destroyAllWindows()
        prior = {'x':x,'y':y}
        
        # Height and width of frame
        N,M,_ = frame.shape
        
        
        # Values for the prior - the bounding box around the can (specified by the user). Top left corner (min_y,min_x). Bottom right corner (max_y,max_x).
        # A width of max_x - min_x and height of max_y - min_y. The particles will have the same width and height.
        min_x = x
        max_x = x+w
        min_y = y
        max_y = y+h

        # Initialize particles
        particles = []
        num_particles = 100
        w = 1/num_particles
        for i in range(num_particles):
            particle_x = float('inf')
            particle_y = float('inf')
            # Generate particles (top left corner coordinates) around the initial object position by sampling a Gaussian (Particles 
            # should be generated inside the boundaries of the image).
            while particle_x > M - (max_x-min_x):
              particle_x = np.random.normal(min_x,3) 
            while particle_y > N - (max_y-min_y):
              particle_y = np.random.normal(min_y,3)
            
            particle_x = round(particle_x)
            particle_y = round(particle_y)
            particles.append({"x":particle_x, "y":particle_y, "w":w})

       # Draw particles.
        for particle in particles:
            particle_x, particle_y, _ = particle.values()
            cv.rectangle(frame, (particle_x, particle_y), (particle_x+(max_x-min_x), particle_y+(max_y-min_y)),(0,255,0),1)
        
        out.write(frame)
    
    else:
        # Resampling
        
        # The resamping is executed in an indirect way, through the resampling of the indices of the particles inside the particles list.
        indices = np.arange(len(particles))
        probabilities = []

        for particle in particles:
            _,_,p = particle.values()
            probabilities.append(p)
            
        
        import copy
        sampled_indices = np.random.choice(indices, size=100, replace=True, p=probabilities)
        # Must use deep copies because list comprehensions with mutatable objects can create reference related problems.
        sampled_particles = [copy.deepcopy(particles[i]) for i in sampled_indices]   #potential duplicates due to bootstrap
        # Deep copy of the list and not a typical copy because the list contains multi level objects and reference problems might arise.
        particles = copy.deepcopy(sampled_particles)
        
        

        # Due to the nature of the resampling stage. After a small number of frames the initial number of particles
        # drops by a significicant amount. Strong particles get resampled resulting in duplicate instances. In order to deal with the
        # problem of particle degeneracy, after each resampling stage, we randomly shift each particle around its original position.
        # In that way, even in the most extreme case where all particles are the same, after the shift, we end up with unique particles.
                        

        # Particle spread. Each particle is shifted in a different way.
        for particle in particles:
            sample = np.random.multivariate_normal([0,0], [[1,0], [0,1]])
            particle['x'] += sample[0]
            particle['y'] += sample[1]
     

        # Prediction - Common for all particles.

        # Gaussian prediction model with zero mean. The std of y is typically smaller than x. The can of cola mostly moves along the x axis.
        sample = np.random.multivariate_normal([0,0], [[2,0],[0,1]])
        x_displacement = sample[0]
        y_displacement = sample[1]
        
        # Update particle position.
        for particle in particles:
            particle_x, particle_y, _ = particle.values()
            particle['x'] = round(particle_x + x_displacement)
            particle['y'] = round(particle_y + y_displacement)


        # Observation. 

        # During the observation staged, each particle receives a weight (probability) that describes the likelihood that the specific
        # particle is estimating the correct position of the object. The observation model is based on edge detection.
    
        edges = cv.Canny(frame, threshold1=30, threshold2=90)
        from scipy import ndimage
        distances_matrix = []
        a=2
        edges_distance_trasformed = ndimage.distance_transform_edt(edges==0)
        for particle in particles:
            #check to see if the particle is out of the frame. If it is, return it in the boundaries.This ensures that there are no errors when indexing in order to obtain the submatrix
            particle_x, particle_y,_ = particle.values()
            if particle_x < 0:
                particle_x = 0
            if particle_x+(max_x-min_x) > edges_distance_trasformed.shape[1]:
                particle_x = particle_x - ( ( particle_x+(max_x-min_x) ) - edges_distance_trasformed.shape[1] )
            if particle_y < 0:
                particle_y = 0
            if particle_y+(max_y-min_y) > edges_distance_trasformed.shape[0]:
                particle_y =  particle_y - ( ( particle_y+(max_y-min_y) ) - edges_distance_trasformed.shape[0] )

            submatrix = edges_distance_trasformed[particle_y : particle_y+(max_y-min_y), particle_x : particle_x+(max_x-min_x)]
            row,col = submatrix.shape

            # Mean distance between each particle and the nearest edges.

            d = np.sum(submatrix) / (row*col)   #divide by the total number of pixel inside the submatrix.

            e_d = np.exp(-a*d)
            distances_matrix.append(e_d)
        
        # Normalize to receive probabilities - particle weights
        probabilities = distances_matrix / np.sum(distances_matrix)
        

        
        # Update particle weights.
        for i, probability in enumerate(probabilities):
            particles[i]['w'] = probability

        # Define the prior - the particle with the biggest weight which will act as the prior for the next run.
        # Get the tuple with the maximum value based on the third element
        prior = max(particles, key=lambda x: x['w']) #dictionary {'x':x,'y':y,'w':w} where x and y the coordinates of the top left corner point and w the weight
        
        # Draw the particles.
        for particle in particles:
            particle_x, particle_y, _ = particle.values()
            cv.rectangle(frame, (particle_x, particle_y), (particle_x+(max_x-min_x), particle_y+(max_y-min_y)),(0,255,0),1)
        #cv.rectangle(frame, (prior['x'], prior['y']),(prior['x']+(max_x-min_x), prior['y']+(max_y-min_y)),(0,255,0),1)
           
        out.write(frame)



cap.release()
out.release()
cv.destroyAllWindows()

      
    


                    

            


        


  


    
        







        


   