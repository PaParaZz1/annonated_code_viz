const express = require("express");
var app = express();
app.use(express.static("./")).listen(5500);
