const exp = require('express')
const app = exp();

app.get('/', (req,res) => {
    res.send('Welcome to Express server!!')
})

app.listen(3000, () => {
    console.log('Listening on port 3000')
})


