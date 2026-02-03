
import lxml.etree as ET  # Faster XML parsing
import glob
import sys

alto_ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def main(files):
	for file in glob.glob(files):
		basename = file.split("/")[-1].split(".")[0]
		parsed = ET.parse(file)
		file_name = parsed.xpath("//alto:sourceImageInformation/alto:fileName", namespaces = alto_ns)[0]
		print(file_name)
		print(basename)
		corresponding_image = f"{basename}.jpg"
		file_name.tag = corresponding_image
		with open(file.replace(".xml", ".clean.xml"), "w") as output_file:
			output_file.write(ET.tostring(parsed).decode())


if __name__ == '__main__':
	main(sys.argv[1])