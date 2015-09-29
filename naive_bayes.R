#Assignment http://classes.engr.oregonstate.edu/eecs/spring2012/cs331/assignments/programming3/programming3.html
#train_naive is concatenation of training feature and label provided 
fortune_cookies<-read.csv('train_naive.txt')
fortune_test<-read.csv('test.txt')
library(tm)
library(SnowballC)

fortune_cookies_all<-rbind(fortune_cookies,fortune_test)
myCorpus <- Corpus(VectorSource(fortune_cookies_all$sentence))
myCorpus[[12]]$content
myCorpus = tm_map(myCorpus, content_transformer(tolower), lazy=TRUE)
myCorpus[[1]]$content
myCorpus = tm_map(myCorpus, PlainTextDocument, lazy=TRUE)
myCorpus[[2]]$content
myCorpus = tm_map(myCorpus, removePunctuation, lazy=TRUE)
myCorpus[[3]]$content
myCorpus <- tm_map(myCorpus, stripWhitespace)
myCorpus[[3]]$content
library(wordcloud)
myCorpus[[3]]$content
sw=read.csv('stoplist.txt')
s=as.character(unlist(sw))
sw1=stopwords('english')
myCorpus = tm_map(myCorpus, removeWords, s, lazy=TRUE)
myCorpus[[9]]$content
myCorpus = tm_map(myCorpus, stemDocument, lazy=TRUE)

myCorpus[[9]]$content

myCorpus=tm_map(myCorpus,stripWhitespace,lazy=TRUE)

library(wordcloud)
wordcloud(myCorpus,min.freq=5,random.order=FALSE)
dtm = DocumentTermMatrix(myCorpus)
inspect(dtm[1:5,1:20])
#find terms that are pretty frequent in dtm i.e atleast appear n times
findFreqTerms(dtm, lowfreq=10)

sparse = removeSparseTerms(dtm, 0.99)
sparse
DescriptionWords = as.data.frame(as.matrix(sparse))
colnames(DescriptionWords)=make.names(colnames(DescriptionWords))
convert_counts <- function(x) {
  x <- ifelse(x > 0, 'yes', 'no')
  x<-as.factor(x)
  return(x)
}
DescriptionWords <- as.data.frame(sapply(DescriptionWords,  convert_counts))

DescriptionWords$label<-fortune_cookies_all$X1

DescriptionWords_train<-head(DescriptionWords,nrow(fortune_cookies))
DescriptionWords_test<-tail(DescriptionWords,nrow(fortune_test))
library(caTools)
set.seed(1234)
split=sample.split(DescriptionWords_train$label,0.7)
train=subset(DescriptionWords_train,split==TRUE)
test=subset(DescriptionWords_train,split!=TRUE)
train$label
test$label


library(e1071)
m1=train
m1$label=NULL
model=naiveBayes(m1,as.factor(train$label),laplace=1)
pred=predict(model,test,type='class')

library(gmodels)

CrossTable(pred, test$label,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))



pred_test=predict(model,DescriptionWords_test,type='class')
table(pred_test,DescriptionWords_test$label)

