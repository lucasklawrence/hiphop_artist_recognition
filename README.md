# hiphop_artist_recognition

This was a project I did for a pattern recognition class at University of Miami.

I used HMMGMMs for a limited vocabulary of hip-hop artists and the word play.  

Loading training data  
Done loading training data

Training the word HMMs  
Done training the word HMMs

Loading Testing Data  
Done Loading Testing Data

Predicting Testing Data  
Done Predicting Testing Data

Overall classification rate is 0.846153846154

The following table has the misclassified examples real and predicted values:

<pre>

+----------------+------------+-----------------+
| Example Number | Real Value | Predicted Value |
+----------------+------------+-----------------+
|       1        |    Play    |      Wayne      |
|       2        |    Play    |      Wayne      |
|       3        |    Play    |     Gambino     |
|       4        |    Play    |      Wayne      |
|       5        |    Play    |     Gambino     |
|       9        |   Drake    |     Gambino     |
|       10       |   Drake    |     Kendrick    |
|       12       |  Kendrick  |     Gambino     |
|       35       |   Snoop    |    PostMalone   |
|       56       |   Wayne    |    MacMiller    |
+----------------+------------+-----------------+

</pre>
Process finished with exit code 0
