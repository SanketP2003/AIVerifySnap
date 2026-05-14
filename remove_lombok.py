import os
import re

files_to_process = [
    "backend/src/main/java/com/backend/aiverifysnap/dto/UserDto.java",
    "backend/src/main/java/com/backend/aiverifysnap/dto/UserCreationDto.java",
    "backend/src/main/java/com/backend/aiverifysnap/model/Users.java",
    "backend/src/main/java/com/backend/aiverifysnap/model/SystemLogs.java",
    "backend/src/main/java/com/backend/aiverifysnap/model/DetectionHistory.java"
]

def capitalize(s):
    return s[0].upper() + s[1:]

for filepath in files_to_process:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove imports
    content = re.sub(r'import lombok\..*?;\n', '', content)
    
    # Remove annotations
    content = re.sub(r'@Data\n|@Getter\n|@Setter\n|@NoArgsConstructor\n|@AllArgsConstructor\n', '', content)
    
    # Find all fields
    fields = re.findall(r'private\s+([\w<>]+)\s+(\w+)\s*;', content)
    
    getters_setters = ""
    for ftype, fname in fields:
        # getter
        getters_setters += f"\n    public {ftype} get{capitalize(fname)}() {{\n        return {fname};\n    }}\n"
        # setter
        getters_setters += f"\n    public void set{capitalize(fname)}({ftype} {fname}) {{\n        this.{fname} = {fname};\n    }}\n"
        
    class_name = re.search(r'public class (\w+)', content).group(1)
    
    constructors = ""
    constructors += f"\n    public {class_name}() {{}}\n"
    if "Dto" in class_name:
        args = ", ".join([f"{ftype} {fname}" for ftype, fname in fields])
        assignments = "".join([f"\n        this.{fname} = {fname};" for _, fname in fields])
        constructors += f"\n    public {class_name}({args}) {{{assignments}\n    }}\n"
        
    # Insert before the last closing brace
    last_brace_idx = content.rfind('}')
    new_content = content[:last_brace_idx] + constructors + getters_setters + "\n}"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

print("Lombok removed successfully!")
