import * as React from "react";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";
import "./home.css";
import TextareaAutosize from "@mui/material/TextareaAutosize";
import Button from "@mui/material/Button";
import { useState } from "react";
import axios from 'axios';

function Home() {
  const [model, setModel] = React.useState("");
  const [input, setInput] = React.useState("");
  const [result, setResult] = React.useState("null");
  const route = "http://127.0.0.1:8000/predict/"

  const handleChange = (event) => {
    setModel(event.target.value);
  };
  const submit = () =>{
    axios.get(route, {
      params: {
        model: model,
        q:input
      }
    })
    .then(res=> {
      setResult(res.data)
    })
    .catch(err=>console.log(err))

  }

  return (
    <div className="main">
      <div className="elm1">
        <FormControl sx={{ minWidth: 200 }}>
          <InputLabel id="demo-simple-select-label">Model</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={model}
            label="Age"
            onChange={handleChange}
          >
            <MenuItem value={"KNN"}>KNN</MenuItem>
            <MenuItem value={"ANN"}>ANN</MenuItem>
            <MenuItem value={"CNN"}>CNN</MenuItem>
            <MenuItem value={"Bert"}>Bert</MenuItem>
          </Select>
        </FormControl>
      </div>
      <div className="elm2">
        <TextareaAutosize
          maxRows={20}
          value={input}
          aria-label="maximum height"
          placeholder="Enter your comment..."
          onChange={e=>setInput(e.target.value)}
          style={{ width: 800, height: 200 }}
        />
        <br></br>
        <Button variant="contained" onClick={submit}>Submit</Button>
      </div>
      <div className="elm3">
        <h2 className="lb">Result:</h2>
        {/* <br></br> */}
        <h2 className="res">{result}</h2>
      </div>
    </div>
  );
}
export default Home;
