
const express = require('express');
const bodyParser = require("body-parser");
const path = require("path");
const cors = require("cors");
require("dotenv").config();
const port = process.env.PORT || 5000;

const cyberRouter = require('./routes/tensorflow.js');
const cyberDataRouter = require('./routes/database.js');

const app = express();

// Middleware
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(cors());
app.use(cookieParser());

// Routes
app.use('/api/cyberAlly', cyberRouter);
app.use('/api/cyberAllyData', cyberDataRouter);


// app.use(express.static(path.join("dist")));

// app.get("*", (req, res) => {
// 	res.sendFile(path.join("dist", "index.html"), {root: path.join(__dirname, "..", "..")});
// });

app.listen(port, () => console.log(`Listening on port ${port}!`));
