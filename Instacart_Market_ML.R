#################
#### Setting ####
#################
options(stringsAsFactors = FALSE, 
        scipen = 100,        # 100자리까지 표시
        max.print = 999999)   
rm(list = ls(all.names = TRUE))
gc(reset = T)
# load("instaCart.RData")
# save.image('instaCart.RData')

## 시스템 사양 확인
# NCmisc::top()$RAM
# parallel::detectCores()

######################
#### load Library ####
######################
library <- c("DescTools", "caret", "tidyverse", "gridExtra", "corrplot", "data.table", "readr", "tibble")
sapply(library, require, character.only = T)

#####################
#### Data upload ####
#####################
list.files(path="../input/", pattern = NULL)

aisles      <- as.tibble(fread('../input/aisles.csv'))
departments <- as.tibble(fread('../input/departments.csv'))
order_Prior <- as.tibble(fread('../input/order_products__prior.csv'))
order_Train <- as.tibble(fread('../input/order_products__train.csv'))
orders      <- as.tibble(fread('../input/orders.csv'))
products    <- as.tibble(fread('../input/products.csv'))
test        <- as.tibble(fread('../input/sample_submission.csv'))

##########################
#### Data description ####
##########################

# print('--------------------aisles------------------------')
# glimpse(aisles)
# print('----------------------departments----------------------')
# glimpse(departments)
# print('------------------------order_Prior--------------------')
# glimpse(order_Prior)
# print('-----------------------order_Train---------------------')
# glimpse(order_Train)
# print('----------------------orders----------------------')
# glimpse(orders)
# print('----------------------products----------------------')
# glimpse(products)
# print('--------------------test------------------------')
# glimpse(test)

####################
#### Data merge ####
####################
products <- products %>% 
  left_join(aisles, by = 'aisle_id') %>% 
  left_join(departments, by = 'department_id')
rm(aisles); rm(departments)

order_Train$user_id <- orders$user_id[match(order_Train$order_id, orders$order_id)]
order_Prior_MST <- order_Prior %>% 
  left_join(orders, by = 'order_id')


##################
#### products ####
##################
prd <- order_Prior_MST %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(user_Prod_CNT = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    prod_CNT = n(),
    prod_Reorder = sum(reordered),
    prod_First_Order = sum(user_Prod_CNT == 1),
    prod_Second_Order = sum(user_Prod_CNT == 2)
  )

prd$prod_Reorder_Prob  <- prd$prod_Second_Order / prd$prod_First_Order
prd$prod_Reorder_Times <- 1 + prd$prod_Reorder / prd$prod_First_Order
prd$prod_Reorder_Ratio <- prd$prod_Reorder / prd$prod_CNT

prd <- prd %>% select(-prod_CNT, -prod_First_Order, -prod_Second_Order)




###############
#### users ####
###############
users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    user_Order = max(order_number),
    user_Period = sum(days_since_prior_order, na.rm = T),
    user_AVG_Day = mean(days_since_prior_order, na.rm = T)
  )

us <- order_Prior_MST %>%
  group_by(user_id) %>%
  summarise(
    user_Total_Prod = n(),
    user_Reorder_Ratio = sum(reordered == 1) / sum(order_number > 1),
    user_Uniq_Prod = n_distinct(product_id)
  )

users <- users %>% inner_join(us)
users$user_AVG_Basket <- users$user_Total_Prod / users$user_Order

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)
rm(us)



##################
#### Database ####
##################
data <- order_Prior_MST %>%
  group_by(user_id, product_id) %>% 
  summarise(
    user_Prod_Order = n(),
    user_Prod_First_Order = min(order_number),
    user_Prod_Last_Order = max(order_number),
    user_Prod_AVG_Cart = mean(add_to_cart_order))

rm(order_Prior_MST, orders)

data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id")

data$user_Prod_Ratio                         <- data$user_Prod_Order / data$user_Order
data$user_Prod_Order_Since_Last_Order        <- data$user_Order - data$user_Prod_Last_Order
data$user_Prod_Order_Ratio_Since_First_Order <- data$user_Prod_Order / (data$user_Order - data$user_Prod_First_Order + 1) 

data <- data %>% 
  left_join(order_Train %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(order_Train, prd, users)

# data %>% glimpse()
# data %>% filter(eval_set == 'train') %>% NROW()  # 8,474,661
# data %>% filter(eval_set == 'test') %>% NROW()   # 4,833,292


##########################
#### Train / Test set ####
##########################

train <- as.data.frame(data[data$eval_set == "train",])

train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0
train$reordered <- as.factor(train$reordered)

train_idx <- createDataPartition(c(train$reordered), p = 0.0001)$Resample1
train <- train[train_idx, ]
rm(train_idx)

test <- as.data.frame(data[data$eval_set == "test",])
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL

rm(data)
rm(order_Prior)
rm(products)



##########################
#### Data Explanation ####
##########################

# print('-----------------------------------------------------')
# train %>% glimpse()
# train %>% Desc()


###################
#### Fit Model ####
###################
library(xgboost)
library(h2o)
h2o.init(nthreads = 15, max_mem_size = "16g")

varnames <- setdiff(colnames(train), c("user_id","order_id","curr_prod_purchased"))
validIdx <- createDataPartition(c(train$reordered), p = 0.3)$Resample1

system.time(
  xgb <- h2o.xgboost(x = varnames
                     ,y = "reordered"
                     ,training_frame   = as.h2o(train[-validIdx, ])
                     ,validation_frame = as.h2o(train[validIdx, ])
                     ,model_id = "xgb_model_1"
                     ,stopping_rounds = 5
                     ,stopping_tolerance = 1e-5
                     ,stopping_metric = "logloss"
                     ,distribution = "bernoulli"
                     ,score_tree_interval = 10
                     ,learn_rate = 0.1
                     ,ntrees = 300
                     ,subsample = 0.7
                     ,colsample_bytree = 0.7
                     ,tree_method = "hist"
                     ,grow_policy = "lossguide"
                     ,booster = "gbtree"
                     ,gamma = 0.7
  )
)
# h2o.partialPlot(object = xgb, data  = as.h2o(train[-validIdx, ]))
# h2o.varimp_plot(xgb, num_of_features = 30)


####################
#### Prediction ####
####################
# test_Pred <- as.data.frame(h2o.predict(xgb, newdata = as.h2o(test)))

# test$reordered <- (test_Pred$p1 > 0.221270) * 1

# NROW(test)  # 4833292
# idx = 1
# for(i in 1:100) {
#     if(idx > 48784959){
#         eval(parse(text = paste0("pred", i, " <- ", "as.data.frame(h2o.predict(xgb, newdata = as.h2o(test[idx:NROW(test), ])))")))
#        } else {
#         eval(parse(text = paste0("pred", i, " <- ", "as.data.frame(h2o.predict(xgb, newdata = as.h2o(test[idx:idx + 48333, ])))")))
#     }
#     idx = idx + 48333 + 2
# }

# for(i in 1:100){
#     eval(parse(text = paste0("test_Pred = rbind(test_Pred, pred", i, ")")))
#     eval(parse(text = paste0("rm(pred", i, ")")
# }







# test_Pred <- NULL
# for(i in 1:NROW(test) / 10) {
#     tmp_Pred <- h2o.predict(xgb, newdata = as.h2o(test[i, ]))
#     tmp_Pred <- as.data.frame(tmp_Pred)
#     
#     test_Pred <- rbind(test_Pred, tmp_Pred)
# }

# test_Pred <- NULL
# test1 <- test[1:1000000, ]
# test_Pred1 <- as.data.frame(h2o.predict(xgb, newdata = as.h2o(test1)))
# test_Pred  <- rbind(test_Pred, test_Pred1)
# rm(test1)
# rm(test_Pred1)
# 
# test2 <- test[1000001:2000000, ]
# test_Pred2 <- as.data.frame(h2o.predict(xgb, newdata = as.h2o(test2)))
# test_Pred  <- rbind(test_Pred, test_Pred2)
# rm(test2)
# rm(test_Pred2)
# 
# test3 <- test[2000001:3000000, ]
# test_Pred3 <- as.data.frame(h2o.predict(xgb, newdata = as.h2o(test3)))
# test_Pred  <- rbind(test_Pred, test_Pred3)
# rm(test3)
# rm(test_Pred3)
# 
# test4 <- test[3000001:4000000, ]
# test_Pred4 <- as.data.frame(h2o.predict(xgb, newdata = as.h2o(test4)))
# test_Pred  <- rbind(test_Pred, test_Pred4)
# rm(test4)
# rm(test_Pred4)
# 
# test5 <- test[4000001:NROW(test), ]
# test_Pred5 <- as.data.frame(h2o.predict(xgb, newdata = as.h2o(test5)))
# test_Pred  <- rbind(test_Pred, test_Pred5)
# rm(test5)
# rm(test_Pred5)
# 
# threshold <- xgb@model$validation_metrics@metrics$max_criteria_and_metric_scores[1, 2]
# test$reordered <- (test_Pred$p1 > threshold) * 1

# 모델 훈련
rf <- h2o.randomForest(x = varnames
                       ,y = "reordered"
                       ,training_frame   = as.h2o(train[-validIdx, ])
                       ,validation_frame = as.h2o(train[validIdx, ])
                       ,nfolds    = 3,
                       ,ntrees    = 10000,
                       ,max_depth = 30,
                       ,keep_cross_validation_predictions = FALSE,
                       ,stopping_rounds = 5,
                       ,stopping_metric = 'AUC'
                       ,stopping_tolerance = 1e-5)

test_Pred <- NULL
test1 <- test[1:1000000, ]
test_Pred1 <- as.data.frame(h2o.predict(rf, newdata = as.h2o(test1)))
test_Pred  <- rbind(test_Pred, test_Pred1)
rm(test1)
rm(test_Pred1)

test2 <- test[1000001:2000000, ]
test_Pred2 <- as.data.frame(h2o.predict(rf, newdata = as.h2o(test2)))
test_Pred  <- rbind(test_Pred, test_Pred2)
rm(test2)
rm(test_Pred2)

test3 <- test[2000001:3000000, ]
test_Pred3 <- as.data.frame(h2o.predict(rf, newdata = as.h2o(test3)))
test_Pred  <- rbind(test_Pred, test_Pred3)
rm(test3)
rm(test_Pred3)

test4 <- test[3000001:4000000, ]
test_Pred4 <- as.data.frame(h2o.predict(rf, newdata = as.h2o(test4)))
test_Pred  <- rbind(test_Pred, test_Pred4)
rm(test4)
rm(test_Pred4)

test5 <- test[4000001:NROW(test), ]
test_Pred5 <- as.data.frame(h2o.predict(rf, newdata = as.h2o(test5)))
test_Pred  <- rbind(test_Pred, test_Pred5)
rm(test5)
rm(test_Pred5)

threshold <- 0.25
test$reordered <- (test_Pred$p1 > threshold) * 1


submission <- test %>%
  filter(reordered == 1) %>%
  arrange(order_id, product_id) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit.csv", row.names = F)







































