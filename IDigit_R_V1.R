library("OpenImageR")
library("imager")
library('pracma')
library('klaR')
library('h2o')
#library('SparkR')


MNIST_train <- file("C:/Users/xinzh/Google Drive/U-I/SP2018Sem/Courses/CS498/Homeworks/Assignment1/data/MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte",'rb')
MNIST_train_label <- file("C:/Users/xinzh/Google Drive/U-I/SP2018Sem/Courses/CS498/Homeworks/Assignment1/data/MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte",'rb')
MNIST_test <- file("C:/Users/xinzh/Google Drive/U-I/SP2018Sem/Courses/CS498/Homeworks/Assignment1/data/MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",'rb')
MNIST_test_label <- file("C:/Users/xinzh/Google Drive/U-I/SP2018Sem/Courses/CS498/Homeworks/Assignment1/data/MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte",'rb')

readBin(MNIST_train,integer(),n = 4,endian = 'big')[2]
readBin(MNIST_train_label,integer(),n = 2,endian = 'big')
readBin(MNIST_test,integer(),n = 4,endian = 'big')
readBin(MNIST_test_label,integer(),n = 2,endian = 'big')

# This is used for regroup plots
par(mfrow=c(4,4))
par(mar=c(0,0,0,0))

# This function is used to cut four edges of MNIST image and recenter digits to 20X20 pixels.
cropper <- function(im) {
  temp<-im
  for (rot_n in 1:4) {
    rot_temp<-0
    for (j in 1:dim(temp)[2]) {
      for (i in 1:dim(temp)[1]) {
        if (temp[i,j]!=0) {
          temp <- temp[,-1:-(j-1)]
          # print(j)
          temp <- rot90(temp)
          rot_temp<-1
          break
        }
      }
      if (rot_temp==1) {
        break
      }
    }
  }
  
  return(resizeImage(temp,20,20))
}

# This function is used to extract data and labels
extract_data_label <- function(train_, label_) {
  
  #extract headers
  num_row <- readBin(train_,integer(),n = 4,endian = 'big')[2]
  readBin(label_,integer(),n = 2,endian = 'big')
  
  #start to extract data, label
  for (Run_ in 1) {
    tic()
    
    st <- 1
    thetime<-1
    for (loop in 1:(num_row/1000)) {
      train_matrix<-matrix()
      for (i in st) {
        print(i)
        m <- matrix(readBin(train_,integer(), size=1, n=28*28, endian="big"),28,28)
        m_label <- matrix(readBin(label_,integer(), size=1, n=1, endian="big"))
        m <- cropper(m)
        m[m!=0] <- 255
        
        m <- as.vector(m)
        m<-t(m)
        m<-data.frame(m)
        
        train_matrix<-m
        train_label_matrix<-m_label
        
        st <- i + 1
      }
      
      ed<-st+998
      for (i in st:ed) {
        print(paste0(loop," ",i))
        m <- matrix(readBin(train_,integer(), size=1, n=28*28, endian="big"),28,28)
        m_label <- matrix(readBin(label_,integer(), size=1, n=1, endian="big"))
        m <- cropper(m)
        m[m!=0] <- 255
        # print(m)
        image(m[,dim(m)[2]:1])
        
        # # if ((dim(m)[1] > 20) | (dim(m)[2] > 20)) {
        # #   break
        # # }  
        m <- as.vector(m)
        m<-t(m)
        m<-data.frame(m)
        
        # print(m)
        
        train_matrix<-rbind(train_matrix,m)
        train_label_matrix<-rbind(train_label_matrix,m_label)
        # print(train_matrix)
      }
      st<-ed+1
      
      
      if (thetime == 1) {
        whole_train<-train_matrix
        whole_train_label<-train_label_matrix
        thetime <- 2
      } else  {
        whole_train<-rbind(whole_train,train_matrix)
        whole_train_label<-rbind(whole_train_label,train_label_matrix)
      }
      
    }
    
    whole_train_label<-factor(whole_train_label)
    whole_<-cbind(whole_train,whole_train_label)
    toc()
  }
  return(whole_)
}

# Extract train's and test's data and labels
train_MNIST <- extract_data_label(MNIST_train, MNIST_train_label)
test_MNIST <- extract_data_label(MNIST_test, MNIST_test_label)

# Fit a NB model
model_NB <- NaiveBayes(train_MNIST$whole_train_label ~ .,data = train_MNIST)

# Predict by NB model
pred_k<-predict(model_NB,test_MNIST)
table(pred_k$class,test_MNIST$whole_train_label)
mean(pred_k$class==test_MNIST$whole_train_label)

# FIT RANDOM FOREST

##initial h2o environment
h2o.init(nthreads=-1,max_mem_size = "2G")

##convert dataframe to h2oframe
train_MNIST.hex <- as.h2o(train_MNIST)
test_MNIST.hex <- as.h2o(test_MNIST)

##predict by trees = 10, 20, 30 and depth = 4, 8, 16
for (trees in seq(10,30,10)){
  for (depth in c(4,8,16)) {
    model_RF <- h2o.randomForest(training_frame = train_MNIST.hex, ntrees = trees, max_depth = depth, x=1:400, y=401)
    pred_rf<-predict(model_RF,test_MNIST.hex)
    pred_rf <- as.data.frame(pred_rf)
    # table(pred_rf$predict,test_MNIST$whole_train_label)
    print(paste0('This is accuracy for ntrees:', trees, ', max_depth:', depth, '. The accuracy is: ', mean(pred_rf$predict==test_MNIST$whole_train_label)))
  }
}



model_RF <- h2o.randomForest(training_frame = train_MNIST, validation_frame = train_MNIST$whole_train_label, ntrees = 10, max_depth = 4, x=1:400, y=401)
h2o.randomForest()
# BACK UP CODE
for (Run_ in 1) {
  tic()
  
  st <- 1
  thetime<-1
  for (loop in 1:4) {
    train_matrix<-matrix()
    for (i in st) {
      print(i)
      m <- matrix(readBin(MNIST_train,integer(), size=1, n=28*28, endian="big"),28,28)
      m_label <- matrix(readBin(MNIST_train_label,integer(), size=1, n=1, endian="big"))
      m <- cropper(m)
      m[m!=0] <- 255
      
      m <- as.vector(m)
      m<-t(m)
      m<-data.frame(m)
      
      train_matrix<-m
      train_label_matrix<-m_label
      
      st <- i + 1
    }
    
    ed<-st+998
    for (i in st:ed) {
      print(paste0(loop," ",i))
      m <- matrix(readBin(MNIST_train,integer(), size=1, n=28*28, endian="big"),28,28)
      m_label <- matrix(readBin(MNIST_train_label,integer(), size=1, n=1, endian="big"))
      m <- cropper(m)
      m[m!=0] <- 255
      # print(m)
      # image(m[,dim(m)[2]:1])
      
      # # if ((dim(m)[1] > 20) | (dim(m)[2] > 20)) {
      # #   break
      # # }  
      m <- as.vector(m)
      m<-t(m)
      m<-data.frame(m)
      
      # print(m)
      
      train_matrix<-rbind(train_matrix,m)
      train_label_matrix<-rbind(train_label_matrix,m_label)
      # print(train_matrix)
    }
    st<-ed+1
    
    
    if (thetime == 1) {
      whole_train<-train_matrix
      whole_train_label<-train_label_matrix
      thetime <- 2
    } else  {
      whole_train<-rbind(whole_train,train_matrix)
      whole_train_label<-rbind(whole_train_label,train_label_matrix)
    }
    
  }
  
  whole_train_label<-factor(whole_train_label)
  whole_<-cbind(whole_train,whole_train_label)
  toc()
}

