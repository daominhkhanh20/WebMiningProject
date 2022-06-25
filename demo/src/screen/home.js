import * as React from "react";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";
import "./home.css";
import TextareaAutosize from "@mui/material/TextareaAutosize";
import Button from "@mui/material/Button";

function Home() {
  const [model, setModel] = React.useState("");
  const [result, setResult] = React.useState("null");

  const handleChange = (event) => {
    setModel(event.target.value);
  };

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
          aria-label="maximum height"
          placeholder="Enter your comment..."
          style={{ width: 800, height: 200 }}
        />
        <br></br>
        <Button variant="contained">Submit</Button>
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
