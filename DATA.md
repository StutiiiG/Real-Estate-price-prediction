# Mumbai1 Dataset Documentation

## Schema

| Column Name           | Type      | Description                                        |
|-----------------------|-----------|----------------------------------------------------|
| Area                  | numeric   | Built-up area in square feet                      |
| No. of Bedrooms       | integer   | Number of bedrooms                                |
| New/Resale            | integer   | 1 = New property, 0 = Resale                      |
| Gymnasium             | integer   | 1 = Gym available                                 |
| Lift Available        | integer   | 1 = Lift available                                |
| Car Parking           | integer   | 1 = Car parking included                          |
| Maintenance Staff     | integer   | 1 = Maintenance staff provided                    |
| 24x7 Security         | integer   | 1 = Security staff available                      |
| Children's Play Area  | integer   | 1 = Dedicated play area present                   |
| Clubhouse             | integer   | 1 = Clubhouse available                           |
| Intercom              | integer   | 1 = Intercom facility                             |
| Landscaped Gardens    | integer   | 1 = Landscaped gardens present                    |
| Indoor Games          | integer   | 1 = Indoor games facilities                       |
| Gas Connection        | integer   | 1 = Piped gas connection                          |
| Jogging Track         | integer   | 1 = Jogging track available                       |
| Swimming Pool         | integer   | 1 = Swimming pool available                       |
| Location              | category  | Locality / area within Mumbai                     |
| Price                 | numeric   | Target variable: property price in INR            |

## Target

The target variable used for modelling is **Price**.

## Preprocessing

- Dropped unnamed index-like columns (`Unnamed: 0`, etc.).
- Handled categorical variables via one-hot encoding in the model pipeline.
  
