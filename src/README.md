<h2>Tips & tricks</h2>

You can use the resample method to group the date:
`df_temp = df['AvgTemperature'].resample('S').mean()`
Commonly used Frequencies and Offsets:
D 	Calendar day 
B 	Business day
W 	Weekly 		
M 	Month end 
BM 	Business month end
Q 	Quarter end 
BQ 	Business quarter end
A 	Year end 
BA 	Business year end
H 	Hours 
